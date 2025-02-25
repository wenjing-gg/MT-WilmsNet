import os
import nrrd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, RangeSlider, Button
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Get the absolute path of the current script (inference.py)
current_file = Path(__file__).resolve()

# Calculate the project root directory path
project_root = current_file.parent.parent  # Upper two level directories

# Add the root directory to the Python path
sys.path.append(str(project_root))

# You can now import normally
from model.mt_wilmsnet import SwinUNETRMultiTask

def find_voi_bounds(image_data):
    """
    Select VOI boundaries interactively.
    In the axial view (top-down), the user drags the mouse to select an x,y region.
    The coronal and sagittal views are displayed and updated in real time,
    showing the spatial 3D relationship of the selected region.
    The user can adjust the z-range using a slider; the overlays in the other views update accordingly.
    
    Args:
        image_data (numpy.ndarray): 3D image data with shape (X, Y, Z).
    
    Returns:
        tuple: (min_bounds, max_bounds), where min_bounds and max_bounds are [x, y, z] indices.
               If selection is not confirmed, returns (None, None).
    """
    selection = {'rect': None, 'z_range': (0, image_data.shape[2]-1), 'confirmed': False}

    # Create layout using GridSpec: Axial view on top; Coronal and Sagittal views on second row; controls at the bottom.
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 1])
    
    # Axial view: top row, fixed z slice for region selection
    ax_axial = fig.add_subplot(gs[0, :])
    z_mid = image_data.shape[2] // 2
    axial_img = ax_axial.imshow(image_data[:, :, z_mid], cmap='gray')
    ax_axial.set_title("Axial View: Drag to select the region")

    # Coronal view: second row left
    ax_coronal = fig.add_subplot(gs[1, 0])
    # Sagittal view: second row right
    ax_sagittal = fig.add_subplot(gs[1, 1])
    
    # Default display: use middle slices if no region is selected
    default_mid_y = image_data.shape[1] // 2
    default_mid_x = image_data.shape[0] // 2
    coronal_img = ax_coronal.imshow(image_data[:, default_mid_y, :], cmap='gray', aspect='auto')
    ax_coronal.set_title("Coronal View")
    sagittal_img = ax_sagittal.imshow(image_data[default_mid_x, :, :], cmap='gray', aspect='auto')
    ax_sagittal.set_title("Sagittal View")
    
    axial_rect_patch = None
    coronal_rect = None
    sagittal_rect = None

    def update_views():
        nonlocal coronal_rect, sagittal_rect
        if selection['rect'] is None:
            return
        x_min, y_min, x_max, y_max = selection['rect']
        z_min, z_max = z_slider.val  # slider returns (z_min, z_max)
        # Update coronal view: use the middle y of selected region as slice index.
        mid_y = (y_min + y_max) // 2
        coronal_img.set_data(image_data[:, mid_y, :])
        ax_coronal.set_title(f"Coronal View (y = {mid_y})")
        # Update sagittal view: use the middle x of selected region as slice index.
        mid_x = (x_min + x_max) // 2
        sagittal_img.set_data(image_data[mid_x, :, :])
        ax_sagittal.set_title(f"Sagittal View (x = {mid_x})")
        
        if coronal_rect is not None:
            coronal_rect.remove()
        if sagittal_rect is not None:
            sagittal_rect.remove()
        # In coronal view: image shape is (X, Z); rectangle drawn at (z_min, x_min)
        rect_width = z_max - z_min
        rect_height = x_max - x_min
        coronal_rect = ax_coronal.add_patch(
            Rectangle((z_min, x_min), rect_width, rect_height,
                      edgecolor='r', facecolor='none', lw=2))
        # In sagittal view: image shape is (Y, Z); rectangle drawn at (z_min, y_min)
        rect_width2 = z_max - z_min
        rect_height2 = y_max - y_min
        sagittal_rect = ax_sagittal.add_patch(
            Rectangle((z_min, y_min), rect_width2, rect_height2,
                      edgecolor='r', facecolor='none', lw=2))
        fig.canvas.draw_idle()

    def onselect(eclick, erelease):
        nonlocal axial_rect_patch
        # In the axial view, image shape is (X, Y). Note: imshow maps x->columns, y->rows.
        x_min = int(min(eclick.ydata, erelease.ydata))
        x_max = int(max(eclick.ydata, erelease.ydata))
        y_min = int(min(eclick.xdata, erelease.xdata))
        y_max = int(max(eclick.xdata, erelease.xdata))
        selection['rect'] = (x_min, y_min, x_max, y_max)
        if axial_rect_patch is not None:
            axial_rect_patch.remove()
        axial_rect_patch = ax_axial.add_patch(
            Rectangle((y_min, x_min), y_max - y_min, x_max - x_min,
                      edgecolor='r', facecolor='none', lw=2))
        fig.canvas.draw_idle()
        update_views()

    rect_selector = RectangleSelector(ax_axial, onselect,
                                      useblit=True, button=[1],
                                      minspanx=5, minspany=5, spancoords='pixels',
                                      interactive=True)

    # Create a slider for z-axis range at the bottom.
    slider_ax = plt.axes([0.15, 0.05, 0.55, 0.03])
    z_slider = RangeSlider(slider_ax, 'Z Range', 0, image_data.shape[2]-1,
                           valinit=(max(0, z_mid-10), min(image_data.shape[2]-1, z_mid+10)))
    z_slider.on_changed(lambda val: update_views())

    # Create a "Confirm" button.
    button_ax = plt.axes([0.75, 0.04, 0.1, 0.05])
    confirm_button = Button(button_ax, 'Confirm')
    def on_confirm(event):
        selection['z_range'] = z_slider.val
        selection['confirmed'] = True
        plt.close(fig)
    confirm_button.on_clicked(on_confirm)

    plt.show()

    if not selection['confirmed'] or selection['rect'] is None:
        print("Selection not confirmed or invalid.")
        return None, None

    x_min, y_min, x_max, y_max = selection['rect']
    z_min, z_max = int(z_slider.val[0]), int(z_slider.val[1])
    min_bounds = np.array([x_min, y_min, z_min])
    max_bounds = np.array([x_max, y_max, z_max])
    return min_bounds, max_bounds

def normalize(img, window_min=-100, window_max=200):
    """
    Normalize the image by clipping to a specified window and scaling to [0, 1].
    """
    img = np.array(img, dtype=np.float32)
    img = np.clip(img, window_min, window_max)
    img = (img - window_min) / (window_max - window_min)
    img = np.clip(img, 0, 1)
    return img

def crop_image(image_data, min_bounds, max_bounds, target_size, expansion_factor=1.2, interpolator=sitk.sitkBSpline):
    """
    Crop the image based on VOI boundaries, expand the region, and resample to target size.
    """
    if min_bounds is None or max_bounds is None:
        return image_data

    # Compute VOI center and expanded half size.
    center = [(mn + mx) / 2 for mn, mx in zip(min_bounds, max_bounds)]
    half_size = [(mx - mn) / 2 * expansion_factor for mn, mx in zip(min_bounds, max_bounds)]
    new_min_bounds = [int(max(0, center[i] - half_size[i])) for i in range(3)]
    new_max_bounds = [int(min(image_data.shape[i] - 1, center[i] + half_size[i])) for i in range(3)]

    # Crop the image.
    cropped_image = image_data[
        new_min_bounds[0]:new_max_bounds[0]+1,
        new_min_bounds[1]:new_max_bounds[1]+1,
        new_min_bounds[2]:new_max_bounds[2]+1
    ]

    # Resample using SimpleITK.
    sitk_image = sitk.GetImageFromArray(np.transpose(cropped_image, (2, 1, 0)))  # (Z, Y, X)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2] * (original_size[2] / target_size[2])
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(target_size)
    resample.SetInterpolator(interpolator)
    resampled_image = resample.Execute(sitk_image)
    return np.transpose(sitk.GetArrayFromImage(resampled_image), (2, 1, 0))

def visualize_single_result(image, seg_mask, class_pred, class_prob, save_path=None):
    """
    Visualize the segmentation result on the original image.
    """
    if image.ndim == 3:
        slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx]
        seg_slice = seg_mask[slice_idx]
    else:
        image_slice = image
        seg_slice = seg_mask

    class_prob = class_prob if class_prob > 0.5 else 1 - class_prob

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(seg_slice, cmap='jet', alpha=0.5)
    label_text = "No metastasis" if class_pred == 0 else "Metastasis"
    plt.title(f"Segmentation Result\n{label_text} (Prob: {class_prob:.2f})")
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def process_cropped_image(image, model, device, mean=0.0, std=1.0):
    """
    Process the cropped image through the model for segmentation inference.
    """
    # Add channel dimension if necessary.
    if image.ndim == 3:
        image_proc = np.expand_dims(image, axis=0)
    else:
        image_proc = image
    image_proc = (image_proc - mean) / std
    image_tensor = torch.tensor(image_proc, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, H, W, D)
    with torch.no_grad():
        outputs = model(image_tensor)
        seg_logits, logits_final, _, _, _, _ = outputs
    class_prob = F.softmax(logits_final, dim=1)[0][1].item()
    class_pred = 1 if class_prob >= 0.5 else 0
    seg_pred = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return image.squeeze(), seg_pred, class_pred, class_prob

def main():

    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select NRRD image file",
        filetypes=[("NRRD files", "*.nrrd")]
    )
    if not image_path:
        print("No file selected. Exiting.")
        return

    # Load raw image from the selected file.
    try:
        image_data, header = nrrd.read(image_path)
    except Exception as e:
        print("Error reading image file:", e)
        return

    # Interactive VOI selection.
    min_bounds, max_bounds = find_voi_bounds(image_data)
    if min_bounds is None or max_bounds is None:
        print("Invalid VOI boundaries. Exiting.")
        return

    # Crop and normalize the image.
    target_size = (64, 64, 64)
    cropped_image = crop_image(image_data, min_bounds, max_bounds, target_size, interpolator=sitk.sitkBSpline)
    normalized_image = normalize(cropped_image)

    # Load model and perform inference.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "D:/Desktop/MNv4_MT_fpn/final_model.pth"
    model = SwinUNETRMultiTask(num_classes=2, in_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    raw_img, seg_mask, class_pred, class_prob = process_cropped_image(normalized_image, model, device, mean=0.0, std=1.0)
    
    # Visualize and save the segmentation result.
    result_path = "D:/Desktop/MNv4_MT_fpn/visualization"
    visualize_single_result(raw_img, seg_mask, class_pred, class_prob, save_path=result_path)
    print("Image processing, inference, and visualization completed. Result saved at", result_path)

if __name__ == "__main__":
    main()
