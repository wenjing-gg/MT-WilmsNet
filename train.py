import os
import time
import argparse
from torch.utils.data import DataLoader
import torch
import math
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import MyNRRDDataSet  # Use your custom dataset classes
from model.mt_wilmsnet import SwinUNETRMultiTask as create_model
from torch.amp import GradScaler
from utils.util import train_one_epoch, evaluate

def wait_for_available_gpu():
    """Check if the card is free in turn, and return its ID once it finds a free card"""
    while True:
        for device_id in range(3, 4):  # Check cuda:0 to cuda:3
            # Gets the free and total video memory information of the video card
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            
            if free_mem >= 0.9 * total_mem:  
                print(f"Device cuda:{device_id} is now fully available. Starting training...")
                return device_id  # Returns the ID of the found idle device
            
            else:
                print(f"Device cuda:{device_id} is currently in use. Free memory: {free_mem} bytes, Total memory: {total_mem} bytes.")
        
        # If no free video card is found, wait 60 seconds and check again
        print("No available GPU found. Waiting...")
        time.sleep(60)

class WarmupCosineLR:
    def __init__(self, optimizer, initial_lr, max_lr, min_lr, epochs, warmup_epochs=5):
        """
        Args:
            optimizer (torch.optim.Optimizer): optimizer
            initial_lr (float): indicates the initial learning rate
            max_lr (float): indicates the maximum learning rate
            min_lr (float): minimum learning rate
            epochs (int): Total training rounds
            warmup_epochs (int): indicates the number of warm-up rounds of the learning rate
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_lambda = self._lr_lambda()
    
    def _lr_lambda(self):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear preheating from initial_lr to max_lr
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing from max_lr to min_lr
                cosine_epochs = self.epochs - self.warmup_epochs
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * (epoch - self.warmup_epochs) / cosine_epochs)
                )
            return lr
        return lr_lambda
    
    def step(self, epoch):
        """
        Renewal learning rate
        Args:
            epoch (int): The current training round (starting from 0)
        """
        lr = self.lr_lambda(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def main(args):
    # Wait and select a free video card
    available_device_id = wait_for_available_gpu()
    # Setup device
    device = torch.device(f"cuda:{available_device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} for training.")
    
    # Initializes the hybrid precision gradient scaler
    scaler = GradScaler("cuda:{available_device_id}") if args.use_amp else None
    print(f"Using Mixed Precision Training: {args.use_amp}")

    # Create a directory to save the model
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    # Initializes the TensorBoard writer
    tb_writer = SummaryWriter()

    # Instantiate the training data set
    train_dataset = MyNRRDDataSet(
        root_dir=args.data_path,
        split='train',
        target_shape=(64, 64, 64),  # Modified target shape
        num_augmentations=args.num_augmentations
    )

    # Instantiate validation data sets that do not require data enhancement
    val_dataset = MyNRRDDataSet(
        root_dir=args.data_path,
        split='test',
        target_shape=(64, 64, 64),  # Modified target shape
        num_augmentations=0
    )

    # Define the data loader
    batch_size = args.batch_size
    nw = 4
    print(f'Using {nw} dataloader workers every process')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create a model
    model = create_model(
        img_size=(64, 64, 64),       # Adjust according to the input size
        in_channels=1,               # Input channel number
        num_classes=args.num_classes,    # Number of categories
        feature_size=48,             # Adjust feature size as needed
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        norm_name="instance",
    ).to(device)
    
    print("Model's state_dict keys:")
    for key in model.state_dict().keys():
        print(key)

    # Load pre-training weights (if provided)
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' does not exist."
        checkpoint = torch.load(args.weights, map_location=device)
        
        # Extract weight dictionary
        if 'net' in checkpoint:
            weights_dict = checkpoint['net']
        else:
            weights_dict = checkpoint

        # Remove the 'module.' prefix
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}

        # Remove 'backbone.' prefix (if present)
        weights_dict = {k.replace('backbone.', ''): v for k, v in weights_dict.items()}

        # Delete decoder related keys
        decoder_keys = [k for k in weights_dict.keys() if k.startswith('decoder')]
        for k in decoder_keys:
            del weights_dict[k]

        # Handle the problem that the number of input channels does not match
        conv1_key = 'swinViT.patch_embed.proj.weight'
        if conv1_key in weights_dict and weights_dict[conv1_key].shape[1] != model.swinViT.patch_embed.proj.weight.shape[1]:
            if weights_dict[conv1_key].shape[1] == 3 and model.swinViT.patch_embed.proj.weight.shape[1] == 1:
                # The pre-training weights are averaged over the channel dimensions
                weights_dict[conv1_key] = weights_dict[conv1_key].mean(dim=1, keepdim=True)

        # Load the pre-training weight
        load_info = model.load_state_dict(weights_dict, strict=False)

        # Print load information
        print("Successfully loaded pre-trained weights.")
        print(f"Missing keys: {load_info.missing_keys}")
        print(f"Unexpected keys: {load_info.unexpected_keys}")

        # Calculate the proportion of loaded parameters
        loaded_params = len(model.state_dict()) - len(load_info.missing_keys)
        total_params = len(model.state_dict())
        load_percentage = (loaded_params / total_params) * 100
        print(f"Percentage of loaded weights: {load_percentage:.2f}%")

    if args.freeze_layers:
        # Specify the layer to freeze
        layers_to_freeze = ["swinViT"]

        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False  # Specify the layer to freeze
                print(f"Freezing {name}")
            else:
                param.requires_grad = True  # Keep other layers trainable
                print(f"Training {name}")

    # Definition optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(pg, lr=args.initial_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1E-4)
    # Define the learning rate scheduler
    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        initial_lr=args.initial_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        epochs=args.epochs,
        warmup_epochs=10  # Adjust the number of preheating rounds as needed
    )

    # Training parameter
    num_epochs = args.epochs
    best_val_auc = 0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Train an epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler
        )

        # val
        val_loss, val_acc, val_auc, val_sen, val_spe = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            num_classes=args.num_classes
        )

        # Record to TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # Adjusted learning rate
        scheduler.step(epoch-1)

        # Check if it is the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_val_sen = val_sen
            best_val_spe = val_spe
            torch.save(model.state_dict(), os.path.join("./weights", "best_model.pth"))
            print(f"Best model saved at epoch {epoch} with Val Auc={val_auc:.4f}")
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # Optional: Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

    # Save the final model
    torch.save(model.state_dict(), os.path.join("./weights", "final_model.pth"))
    print("Training complete. Final model saved.")
    # Print and save the best result information
    print(f"\nBest Results at Epoch {best_epoch}:")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Val Sensitivity: {best_val_sen:.4f}")
    print(f"Best Val Specificity: {best_val_spe:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vision Transformer for 3D Classification')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of target classes')
    parser.add_argument('--epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation')
    parser.add_argument('--initial_lr', type=float, default=0, help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=3e-5, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=0, help='Initial learning rate')
    
    # The root directory of the data set
    parser.add_argument('--data_path', type=str, default="/home/yuwenjing/data/Wilms_tumor_training_data", help='Path to the dataset')

    # Pretrain the weight path and set it to null if you don't want to load it
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    
    # Freeze weight or not
    parser.add_argument('--freeze_layers', type=bool, default=True, help='Freeze layers except head and pre_logits')
    
    # Whether to use mixed precision training
    parser.add_argument('--use_amp', action='store_true', default=True ,help='Use Automatic Mixed Precision')
    
    # Data enhancement quantity
    parser.add_argument('--num_augmentations', type=int, default=6, help='Number of augmentations per sample during training')
    
    opt = parser.parse_args()
    
    main(opt)
