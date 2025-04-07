import numpy as np
import tifffile as tiff
import os
import natsort
import cv2 as cv
import scipy.ndimage as ndi
import random
import colorsys

def random_vivid_color():
    h = random.random()                 # Hue: random in [0, 1]
    s = 1.0                             # Full saturation
    v = 1.0                             # Full brightness
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def overlay_contours_3d(cell_directory, original_directory, output_dir, alpha=0.5, threshold=0):
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted file lists
    cell_files = natsort.natsorted(os.listdir(cell_directory))
    original_files = natsort.natsorted(os.listdir(original_directory))

    if len(cell_files) != len(original_files):
        raise ValueError("The input TIFF stacks must have the same number of frames.")

    # Load the entire volume from each directory into 3D arrays
    cell_stack = []
    original_stack = []
    for file in cell_files:
        cell_img = tiff.imread(os.path.join(cell_directory, file))
        cell_stack.append(cell_img)
    for file in original_files:
        orig_img = tiff.imread(os.path.join(original_directory, file))
        original_stack.append(orig_img)

    cell_volume = np.array(cell_stack)
    original_volume = np.array(original_stack)

    # If the cell volume is not binary, threshold it.
    # Here we assume that pixel values greater than threshold indicate cell structures.
    binary_cell_volume = cell_volume > threshold

    # Label the 3D connected components.
    # Using a 3x3x3 structuring element for 26-connectivity.
    structure = np.ones((3, 3, 3))
    labeled_volume, num_features = ndi.label(binary_cell_volume, structure=structure)
    print(f"Found {num_features} connected components in 3D.")
    
    label_to_color = {
        label: random_vivid_color()
        for label in range(1, num_features + 1)
    }

    # Process each slice in the volume.
    for z in range(labeled_volume.shape[0]):
        # Get the original slice and normalize/convert to color if needed.
        orig_slice = original_volume[z]
        if orig_slice.dtype != np.uint8:
            orig_slice = cv.normalize(orig_slice, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        if len(orig_slice.shape) == 2:
            orig_color = cv.cvtColor(orig_slice, cv.COLOR_GRAY2BGR)
        else:
            orig_color = orig_slice.copy()

        # Create an overlay mask (same dimensions as the slice).
        overlay_mask = np.zeros_like(orig_color)

        # Get the segmentation labels for the current slice.
        slice_labels = labeled_volume[z]
        unique_labels = np.unique(slice_labels)
        for label in unique_labels:
            if label == 0:
                continue  # Skip background

            # Create a binary mask for the current component.
            comp_mask = (slice_labels == label).astype(np.uint8) * 255

            # Find contours for the component mask.
            contours, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # Draw filled contours with the segment’s assigned color.
            cv.drawContours(overlay_mask, contours, -1, label_to_color[label], thickness=cv.FILLED)

        # Blend the original image with the colored overlay mask.
        blended = cv.addWeighted(orig_color, 1, overlay_mask, alpha, 0)

        # Save the blended slice.
        tiff.imwrite(os.path.join(output_dir, f"frame_{z:04d}.tif"), blended)

    print(f"3D overlay saved in {output_dir}")

if __name__ == '__main__':
    # Example usage:
    overlay_contours_3d('multi_stage', '9_slices', "contours", alpha=0.3)
