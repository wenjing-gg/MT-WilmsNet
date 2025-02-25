import os
import nrrd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import random
from scipy.ndimage import binary_closing, binary_opening
from skimage import measure
import matplotlib.pyplot as plt

def find_voi_bounds(label_data):
    """
    The boundaries of VOI in label data are determined and connected component analysis
    and morphological operations are applied to improve VOI quality
    
    Args:
        label_data (numpy.ndarray): label data list
    
    Returns:
        tuple: (min_bounds, max_bounds) or (None, None) if there is no VOI area
    """

    # Apply morphological operations: first 'Open' and then 'Close' to remove small noise and fill small holes
    cleaned_label = binary_opening(label_data, structure=np.ones((3, 3, 3))).astype(np.uint8)
    cleaned_label = binary_closing(cleaned_label, structure=np.ones((3, 3, 3))).astype(np.uint8)
    
    # Analysis of connected components, select the largest connected region
    labeled_array, num_features = measure.label(cleaned_label, return_num=True, connectivity=1)
    if num_features == 0:
        return None, None
    
    # Find the largest connected component
    largest_label = 1 + np.argmax([np.sum(labeled_array == i) for i in range(1, num_features+1)])
    voi_mask = (labeled_array == largest_label).astype(np.uint8)
    
    non_zero_indices = np.argwhere(voi_mask)
    if non_zero_indices.size == 0:
        return None, None  # Return (None, None) if there is no VOI region
    min_bounds = non_zero_indices.min(axis=0)
    max_bounds = non_zero_indices.max(axis=0)
    return min_bounds, max_bounds

def normalize(img, window_min=-100, window_max=200):
    """
    Crop the image to the specified window range and normalize it to the [0, 1] range
    
    Args:
        img (numpy.ndarray): input image array shape (X, Y, Z)
        window_min (float): window lower limit (minimum HU value)
        window_max (float): window upper limit (maximum HU value)
    
    Returns:
        numpy.ndarray: normalized to [0, 1]
    """
    img = np.array(img, dtype=np.float32)
    
    # Crop the image to the window range
    img = np.clip(img, window_min, window_max)
    
    # normalized to [0, 1]
    img = (img - window_min) / (window_max - window_min)
    img = np.clip(img, 0, 1)
    return img

def crop_image(image_data, min_bounds, max_bounds, target_size, expansion_factor=1.2, interpolator=sitk.sitkBSpline):
    """
    Crop the image according to the VOI boundary, extend the crop range, and adjust to the target size
    
    Args:
        image_data (numpy.ndarray): original image shape (X, Y, Z)
        min_bounds (array-like): the minimum bounds of VOI (x_min, y_min, z_min)
        max_bounds (array-like): the largest bounds of VOI (x_max, y_max, z_max)
        target_size (tuple): target image size (x, y, z)。
        expansion_factor (float): expansion factor, default is 1.2。
        interpolator (SimpleITK interpolator): interpolator method, default is 'BSpline'
    
    Returns:
        numpy.ndarray: cropped and resampled image data shape (X, Y, Z)。
    """

    # if there is no VOI area, it does not crop and directly returns to the original image
    if min_bounds is None or max_bounds is None:
        return image_data

    # calculate the extended boundary range
    center = [(mn + mx) / 2 for mn, mx in zip(min_bounds, max_bounds)]
    half_size = [(mx - mn) / 2 * expansion_factor for mn, mx in zip(min_bounds, max_bounds)]
    
    # extended min bounds and max bounds
    new_min_bounds = [int(max(0, center[i] - half_size[i])) for i in range(3)]
    new_max_bounds = [int(min(image_data.shape[i] - 1, center[i] + half_size[i])) for i in range(3)]

    # crop the image
    cropped_image = image_data[
        new_min_bounds[0]:new_max_bounds[0]+1,
        new_min_bounds[1]:new_max_bounds[1]+1,
        new_min_bounds[2]:new_max_bounds[2]+1
    ]

    # resampling with SimpleITK
    sitk_image = sitk.GetImageFromArray(np.transpose(cropped_image, (2, 1, 0)))  # 转换为 (Z, Y, X)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    # calculate the new spatial resolution
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2] * (original_size[2] / target_size[2])
    ]
    
    # set the resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(target_size)
    resample.SetInterpolator(interpolator)
    
    resampled_image = resample.Execute(sitk_image)

    return np.transpose(sitk.GetArrayFromImage(resampled_image), (2, 1, 0))

def find_image_label_paths(data_dirs, unique_id):
    """
    Find the image file and tag file path for the given 'unique_id'
    
    Args:
        data_dirs (list): data folder path list
        unique_id (str): unique_id path
    
    Returns:
        tuple: (image_path, label_path) or (None, None) if not found
    """
    base_id = unique_id.replace('_image', '')
    image_path, label_path = None, None

    for data_dir in data_dirs:
        possible_image_path = os.path.join(data_dir, f"{base_id}_image.nrrd")
        possible_label_path = os.path.join(data_dir, f"{base_id}_label.nrrd")

        if os.path.exists(possible_image_path):
            image_path = possible_image_path
        if os.path.exists(possible_label_path):
            label_path = possible_label_path

        if image_path and label_path:
            break

    return image_path, label_path

def ensure_directory_exists(directory):
    """ensure that the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_data_by_label(all_files, split_ratio=0.8):
    """
    Data is divided into train set and test set according to labels, keeping the proportion of labels

    Args:
        all_files (list): a list of all file information, each element is(unique_id, image_path, label_path, label_class, dataset_name)
        split_ratio (float): train set ratio, default is 0.8。
    
    Returns:
        tuple: (train_files, test_files)
    """

    label_0 = [file for file in all_files if file[3] == 0]
    label_1 = [file for file in all_files if file[3] == 1]

    random.shuffle(label_0)
    random.shuffle(label_1)

    # computed split point
    split_0 = int(len(label_0) * split_ratio)
    split_1 = int(len(label_1) * split_ratio)

    # segmentation
    train_files = label_0[:split_0] + label_1[:split_1]
    test_files = label_0[split_0:] + label_1[split_1:]

    random.shuffle(train_files)
    random.shuffle(test_files)

    return train_files, test_files

def process_and_crop_images(all_files, target_size=(64, 64, 64)):
    """
    Process and crop all files and return the processed (dataset_name, unique_id, normalized_image, cropped_label, label_class)
    
    Args:
        all_files (list): a list of all file information, each element is (unique_id, image_path, label_path, label_class, dataset_name)
        target_size (tuple): the target size of cropped image, default is (64, 64, 64)。
    
    Returns:
        list: a list of cropped images, each element is (dataset_name, unique_id, normalized_image, cropped_label, label_class)
    """
    processed_images = []

    for unique_id, image_path, label_path, label_class, dataset_name in tqdm(all_files, desc="Processing images"):
        try:
            label_data, _ = nrrd.read(label_path)
            image_data, _ = nrrd.read(image_path)
        except Exception as e:
            print(f"Error: Failed to read file, ID: {unique_id}, Wrong: {e}")
            continue

        # determine VOI boundaries
        min_bounds, max_bounds = find_voi_bounds(label_data)

        if min_bounds is None or max_bounds is None:
            print(f"Warning: ID {unique_id} No VOI detected. Skip.")
            continue

        cropped_image = crop_image(image_data, min_bounds, max_bounds, target_size, interpolator=sitk.sitkBSpline)
        cropped_label = crop_image(label_data, min_bounds, max_bounds, target_size, interpolator=sitk.sitkNearestNeighbor)

        normalized_image = normalize(cropped_image)

        # make sure the label is binary
        cropped_label = (cropped_label > 0).astype(np.uint8)

        # add to the processed list
        processed_images.append((dataset_name, unique_id, normalized_image, cropped_label, label_class))

    return processed_images

def save_cropped_images(processed_images, train_dir, test_dir, split_ratio=0.8):
    """
    Save the cropped image and label file to the train set and test set folds
    
    Args:
        processed_images (list): indicates the cropped image list, each element is (dataset_name, unique_id, normalized_image, cropped_label, label_class)
        train_dir (str): target directory for train set
        test_dir (str): target directory for test set
        split_ratio (float): train set ratio, default is 0.8
    """

    label_0 = [file for file in processed_images if file[4] == 0]
    label_1 = [file for file in processed_images if file[4] == 1]

    random.shuffle(label_0)
    random.shuffle(label_1)

    # computed split point
    split_0 = int(len(label_0) * split_ratio)
    split_1 = int(len(label_1) * split_ratio)

    # segmentation
    train_files = label_0[:split_0] + label_1[:split_1]
    test_files = label_0[split_0:] + label_1[split_1:]

    random.shuffle(train_files)
    random.shuffle(test_files)

    # save train set
    for dataset_name, unique_id, normalized_image, cropped_label, label_class in tqdm(train_files, desc="Saving training images"):
        subdir = '0' if label_class == 0 else '1'
        output_subdir = os.path.join(train_dir, subdir)
        ensure_directory_exists(output_subdir)
        
        # save image
        output_path_img = os.path.join(output_subdir, f"{dataset_name}_{unique_id}.nrrd")
        # save label
        output_path_label = os.path.join(output_subdir, f"{dataset_name}_{unique_id}_mask.nrrd")
        
        try:
            nrrd.write(output_path_img, normalized_image)
            nrrd.write(output_path_label, cropped_label)
        except Exception as e:
            print(f"Error: Failed to save training image or label, ID: {unique_id}, Wrong: {e}")

    # save test set
    for dataset_name, unique_id, normalized_image, cropped_label, label_class in tqdm(test_files, desc="Saving testing images"):
        subdir = '0' if label_class == 0 else '1'
        output_subdir = os.path.join(test_dir, subdir)
        ensure_directory_exists(output_subdir)
        
        # save image
        output_path_img = os.path.join(output_subdir, f"{dataset_name}_{unique_id}.nrrd")
        # save label
        output_path_label = os.path.join(output_subdir, f"{dataset_name}_{unique_id}_mask.nrrd")

        try:
            nrrd.write(output_path_img, normalized_image)
            nrrd.write(output_path_label, cropped_label)
        except Exception as e:
            print(f"Error: Failed to save test image or label, ID: {unique_id}, Wrong: {e}")

def visualize_voi(image_data, label_data, unique_id, save_dir):
    """
    Visualize and save VOI crop result
    
    Args:
        image_data (numpy.ndarray): normalized image data shape (X, Y, Z)
        label_data (numpy.ndarray): cropped label data shape (X, Y, Z)
        unique_id (str): unique identifier
        save_dir (str): save directory for visual result
    """
    ensure_directory_exists(save_dir)

    # visual intermediate slice
    z_mid = image_data.shape[2] // 2
    y_mid = image_data.shape[1] // 2
    x_mid = image_data.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_data[:, :, z_mid], cmap='gray')
    axes[0].imshow(label_data[:, :, z_mid], cmap='jet', alpha=0.5)
    axes[0].set_title('Axial View')

    axes[1].imshow(image_data[:, y_mid, :], cmap='gray')
    axes[1].imshow(label_data[:, y_mid, :], cmap='jet', alpha=0.5)
    axes[1].set_title('Coronal View')

    axes[2].imshow(image_data[x_mid, :, :], cmap='gray')
    axes[2].imshow(label_data[x_mid, :, :], cmap='jet', alpha=0.5)
    axes[2].set_title('Sagittal View')

    plt.suptitle(f"VOI Visualization for {unique_id}")
    plt.savefig(os.path.join(save_dir, f"{unique_id}_voi_visualization.png"))
    plt.close()

def main():
    """
    main function, performs all data processing steps
    """

    # set random seeds to ensure repeatability
    random.seed(42)
    np.random.seed(42)

    # define the data set configuration
    datasets = [
        {
            'data_dirs': ['/home/yuwenjing/data/Wilms_tumor_CT_data/Data0', '/home/yuwenjing/data/Wilms_tumor_CT_data/Data1'],
            'dataset_name': 'sm1',
        },
        {
            'data_dirs': ['/home/yuwenjing/data/Wilms_tumor_CT_data/Data3-0', '/home/yuwenjing/data/Wilms_tumor_CT_data/Data3-1'],
            'dataset_name': 'sm2',
        }
    ]

    all_files = []
    for dataset in datasets:
        data_dirs = dataset['data_dirs']
        dataset_name = dataset['dataset_name']

        for data_dir in data_dirs:
            # determine labels based on the folder name
            if 'Data0' in data_dir or 'Data3-0' in data_dir:
                label_class = 0
            elif 'Data1' in data_dir or 'Data3-1' in data_dir:
                label_class = 1
            else:
                print(f"Warning: Unknown data directory {data_dir}, skip it.")
                continue

            for filename in os.listdir(data_dir):
                if filename.endswith('_image.nrrd'):
                    unique_id = filename.replace('_image.nrrd', '')
                    image_path, label_path = find_image_label_paths(data_dirs, unique_id)

                    if not image_path or not label_path:
                        print(f"Warning: Image or tag file not found, ID: {unique_id}")
                        continue

                    all_files.append((unique_id, image_path, label_path, label_class, dataset_name))

    print(f"There are {len(all_files)} files to be processed.")

    # process and crop all images
    processed_images = process_and_crop_images(all_files, target_size=(64, 64, 64))
    print(f"Processing complete, ready to save {len(processed_images)} cropped image and label.")

    # define output directory
    output_dir = 'data/Wilms_tumor_training_data'
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    visualization_dir = os.path.join(output_dir, 'visualizations')

    # create train/0, train/1, test/0, test/1 catalogue
    ensure_directory_exists(train_dir)
    ensure_directory_exists(test_dir)
    ensure_directory_exists(os.path.join(train_dir, '0'))
    ensure_directory_exists(os.path.join(train_dir, '1'))
    ensure_directory_exists(os.path.join(test_dir, '0'))
    ensure_directory_exists(os.path.join(test_dir, '1'))
    ensure_directory_exists(visualization_dir)

    # save the cropped image and label to the 'train' / 'test' folder
    save_cropped_images(processed_images, train_dir, test_dir, split_ratio=0.8)
    print("The cropped images and labels are saved to the training set and test set directories.")

    # visual part
    for dataset_name, unique_id, normalized_image, cropped_label, label_class in tqdm(processed_images, desc="Visualizing VOI"):
        visualize_voi(normalized_image, cropped_label, unique_id, visualization_dir)

    print("All VOI visualizations are completed and saved.")
    print("Data processing and division complete!")

if __name__ == "__main__":
    main()