import copy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import MyNRRDDataSet  # Use your custom dataset classes
from grad_cam import GradCAM
from torch import nn
import torch.nn.functional as F
from model.mt_wilmsnet import SwinUNETRMultiTask as create_model


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def visualize_comparison(input_data, feature_map, slice_idx, decoder):
    """3D visualization contrast function"""
    # Get raw slice
    original_slices = {
        'x': input_data[0, 0, slice_idx, :, :].numpy(),
        'y': input_data[0, 0, :, slice_idx, :].numpy(),
        'z': input_data[0, 0, :, :, slice_idx].numpy()
    }

    # Feature map preprocessing
    if isinstance(feature_map, np.ndarray):
        feature_map = torch.tensor(feature_map, dtype=torch.float32)
        feature_map=feature_map.unsqueeze(0)
    # 3D interpolation to original dimensions
    upsampled_feature = F.interpolate(
        feature_map,
        size=tuple(input_data.shape[2:]),
        mode='trilinear',
        align_corners=False
    ).squeeze()

    # Create a visual canvas
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plt.set_cmap('jet')

    # Traverse three axes
    for idx, axis in enumerate(['x', 'y', 'z']):
        # Primary slice
        axes[idx, 0].imshow(original_slices[axis], cmap='gray')
        axes[idx, 0].set_title(f'Original {axis.upper()}-Slice {slice_idx}')
        # Feature map slice
        if axis == 'x':
            feature_slice = upsampled_feature[slice_idx, :, :]
        elif axis == 'y':
            feature_slice = upsampled_feature[:, slice_idx, :]
        else:
            feature_slice = upsampled_feature[:, :, slice_idx]

        # normalization
        feature_slice = (feature_slice - feature_slice.min()) / (feature_slice.max() - feature_slice.min())

        # Overlay visualization
        axes[idx, 1].imshow(original_slices[axis], cmap='gray')
        im = axes[idx, 1].imshow(feature_slice, alpha=0.5)
        plt.colorbar(im, ax=axes[idx, 1])
        axes[idx, 1].set_title(f'Feature Map {axis.upper()}-Slice {slice_idx}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'D:\Desktop\yuwenjing\img\{decoder}slice_idx{slice_idx}.png')
def main():
    model = create_model(
        img_size=(64, 64, 64),  # Adjust according to the input size
        in_channels=1,  # Input channel number
        num_classes=2,  # Number of categories
        feature_size=48,  # Adjust feature size as needed
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        norm_name="instance",
    )
    print(model)
    # Replacement corresponding weight
    weight_path=r"D:\Desktop\MNv4_MT_fpn\pretrain_weight\best_model.pth"
    weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights, strict=False)
    model.eval()


    # load image
    val_dataset = MyNRRDDataSet(
        root_dir="data/Wilms_tumor_training_data/test",
        split='test',
        target_shape=(64, 64, 64),  # Modified target shape
        num_augmentations=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    for batch_idx, batch in enumerate(val_loader):
        imgs, class_labels, seg_labels = batch
        break

    input_data = imgs   # [1,1,64,64,64]


    input_data = torch.tensor(input_data, dtype=torch.float32)

    print(input_data)
    target_layers = [model.decoder1]
    print(target_layers)


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    target_category = 0

    grayscale_cam = cam(input_tensor=input_data, target_category=target_category)

    print(grayscale_cam)
    # Get slices on different axes
    decoder = 'class_0'
    for slice_idx in range(64):
        visualize_comparison(input_data, grayscale_cam, slice_idx, decoder)
if __name__ == '__main__':
    main()
