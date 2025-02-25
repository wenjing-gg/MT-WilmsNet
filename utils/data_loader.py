import os
import re
import nrrd
import random
from monai.transforms import (
    RandFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    ToTensord,
    ScaleIntensityRanged,
    Compose
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MyNRRDDataSet(Dataset):
    """Customize NRRD format datasets, load images and masks, and process images and masks as (64,64,64) shapes"""

    def __init__(self, root_dir: str, split: str, target_shape=(64, 64, 64), num_augmentations=3):
        """
        Args:
            root_dir (str): The root directory of the dataset
            split (str): Data set partitioning, 'train' or 'test'
            target_shape (tuple, optional): Target shape (D, H, W)
            num_augmentations (int): Number of enhanced samples generated per original image (for training purposes only)
        """
        self.split = split
        self.target_shape = target_shape
        self.num_augmentations = num_augmentations  # Use only when training sets

        # Uniform intensity normalization transform (for both training and testing)
        self.intensity_transform = ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        )

        # Only the list of random enhancements used for training (select one at random) is applied to both image and mask
        if split == 'train':
            self.augmentations = [
                RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=1),  # X-axis flip
                RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=1),  # Y-axis flip
                RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=1),  # Z-axis flip
                RandZoomd(keys=["image", "mask"], min_zoom=0.8, max_zoom=1.2, prob=0.8),
                RandGaussianNoised(keys=["image"], mean=0.0, std=0.03, prob=0.8),
                RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), prob=0.8),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.8),
            ]
        else:
            self.augmentations = []

        # Transfer tensor (used for all splits)
        self.to_tensor = ToTensord(keys=["image", "mask"])

        self.data_list = []  # Store image data and labels for all samples

        # Load data
        self._load_images_from_folder(os.path.join(root_dir, split, '0'), label=0)  # NoMetastasis
        self._load_images_from_folder(os.path.join(root_dir, split, '1'), label=1)  # Metastasis

    def _load_images_from_folder(self, folder: str, label: int):
        """Load all NRRD files in the specified folder, assign category labels, and load the corresponding masks"""
        for filename in os.listdir(folder):
            if filename.endswith(".nrrd") and not filename.endswith("_mask.nrrd"):
                img_path = os.path.join(folder, filename)
                mask_filename = filename.replace(".nrrd", "_mask.nrrd")
                mask_path = os.path.join(folder, mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_filename} not found for image {filename}. Skipping.")
                    continue
                # Process images and masks
                img = self._process_nrrd(img_path)
                seg_label = self._process_nrrd(mask_path)
                # Matching ID
                match = re.match(r'(sm\d+)_(\d+)', filename)
                if match:
                    prefix = match.group(1)
                    id_number = match.group(2)
                    id_ = f"{prefix}_{id_number}_image"
                    self.data_list.append((img, label, seg_label))
                else:
                    print(f"Warning: Filename {filename} does not match expected pattern 'sm<digit>_<number>.nrrd'")

    def _process_nrrd(self, file_path):
        """Process the NRRD file and return the adjusted image"""
        data, header = nrrd.read(file_path)
        # Convert (H, W, D) to (D, H, W)
        img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)

        # Make sure the input is 3D data
        if img.ndim != 3:
            raise ValueError(f"Image at {file_path} is not a 3D volume.")

        img = self.interpolate_to_shape(img, self.target_shape)
        return img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label, seg_label = self.data_list[idx]

        if self.split == 'train':
            # Training set: Generate multiple enhanced versions of each sample
            augmented_samples = []
            for _ in range(self.num_augmentations):
                # 1. Add channel dimensions first
                sample_dict = {"image": img.clone().unsqueeze(0), "mask": seg_label.clone().unsqueeze(0)}  # [C, D, H, W]
                
                # 2. Intensity normalization
                sample_dict = self.intensity_transform(sample_dict)

                # 3. Randomly select and apply enhancements
                if self.augmentations:
                    chosen_aug = random.choice(self.augmentations)
                    sample_dict = chosen_aug(sample_dict)

                # 4. Convert to tensor (already a tensor, no need to convert again)
                # sample_dict = self.to_tensor(sample_dict)  # removable

                # 5. Normalized image
                normalized_img = self.normalize(sample_dict["image"])
                
                # 6. Process the mask and ensure that the type is long
                normalized_mask = sample_dict["mask"]
                if normalized_mask.dtype != torch.long:
                    normalized_mask = normalized_mask.long()

                # 7. Add to Enhanced sample list
                augmented_samples.append((normalized_img, label, normalized_mask))
            return augmented_samples
        else:
            # Test set: Only one processing, no random enhancement
            sample_dict = {"image": img.clone().unsqueeze(0), "mask": seg_label.clone().unsqueeze(0)}  # [C, D, H, W]
            sample_dict = self.intensity_transform(sample_dict)
            sample_dict = self.to_tensor(sample_dict)

            # Normalized image
            normalized_img = self.normalize(sample_dict["image"])
            
            # Process the mask and ensure that the type is long
            normalized_mask = sample_dict["mask"]
            if normalized_mask.dtype != torch.long:
                normalized_mask = normalized_mask.long()

            processed_img = normalized_img  # [C, D, H, W]
            processed_mask = normalized_mask  # [C, D, H, W]
            return [(processed_img, label, processed_mask)]

    def normalize(self, img, mean=None, std=None):
        """Standard normalization of input images (zero mean, unit variance)"""
        if mean is None:
            mean = torch.mean(img).item()
        if std is None:
            std = torch.std(img).item()

        if std > 0:
            img = (img - mean) / std
        else:
            img = torch.zeros_like(img)
        return img

    def interpolate_to_shape(self, img, target_shape):
        """Interpolates the input 3D image to the specified shape"""
        current_shape = img.shape
        if current_shape == target_shape:
            # print(f" Image size already matches the target size {target_shape}, no adjustment required.")
            return img
        img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        img = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=True)
        img = img.squeeze(0).squeeze(0)      # (D, H, W)
        return img

    @staticmethod
    def collate_fn(batch):
        # batch is a list where each element is a sample (multiple enhancement samples are included during training)
        # Each enhancement sample is (augmented_img, class_label, seg_label)
        # or one for testing (img, class_label, seg_label)
        all_samples = [sample for sublist in batch for sample in sublist]
        all_imgs, all_class_labels, all_seg_labels = zip(*all_samples)
        all_imgs = torch.stack(all_imgs, dim=0)          # [B_total, C, D, H, W]
        all_class_labels = torch.tensor(all_class_labels, dtype=torch.long)  # [B_total]
        all_seg_labels = torch.stack(all_seg_labels, dim=0)  # [B_total, C_seg, D, H, W]
        return all_imgs, all_class_labels, all_seg_labels

if __name__ == "__main__":
    root_dir = r'/home/yuwenjing/data/Wilms_tumor_training_data'

    # Initializes the training data set
    train_dataset = MyNRRDDataSet(
        root_dir=root_dir,
        split='train',
        target_shape=(64, 64, 64),
        num_augmentations=8
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # Each batch returns num_augmentations samples
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=MyNRRDDataSet.collate_fn
    )
    print(f"Total number of training set samples:{len(train_dataset)}")
    for imgs, labels, seg_labels in train_dataloader:
        print("Training batch information:")
        print(f"Number of images (enhanced sample number) : {imgs.shape[0]}")  # num_augmentations
        print(f"Individual image shapes: {imgs[0].shape}")           # [C, D, H, W]
        print(f"Labels: {labels}")                          # [B_total]
        print(f"Seg labels Shape: {seg_labels.shape}")      # [B_total, C_seg, D, H, W]
        break

    # Initializes the test data set
    test_dataset = MyNRRDDataSet(
        root_dir=root_dir,
        split='test',
        target_shape=(64, 64, 64),
        num_augmentations=1  # Test sets usually do not require multiple enhancements
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Each batch returns one sample
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=MyNRRDDataSet.collate_fn
    )
    print(f"Total number of test set samples: {len(test_dataset)}")
    for imgs, labels, seg_labels in test_dataloader:
        print("Test batch information:")
        print(f"Number of images: {imgs.shape[0]}")               # for 1
        print(f"Individual image shapes: {imgs[0].shape}")         # [C, D, H, W]
        print(f"Labels: {labels}")                          # [1]
        print(f"Seg labels Shape: {seg_labels.shape}")      # [1, C_seg, D, H, W]
        break
