import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
from numba import njit, prange
from math import sqrt
import scipy.ndimage
from skimage import filters, feature, morphology, measure, segmentation, exposure
import natsort
import cv2 as cv
import draw_contours as dc

# Define folder names
foldername = "indiv_slices_10"

def load_data(foldername):
    print("Reading .tiff files ...")
    tiff_files = sorted(
        [f for f in os.listdir(foldername) if f.endswith(".tiff") or f.endswith(".tif")]
    )
    images = []
    for file in tiff_files:
        image_path = os.path.join(foldername, file)
        img = iio.imread(image_path)
        images.append(img)
    image_stack = np.stack(images, axis=0)
    return image_stack


def process_layer(img):
    # 1. Background & contrast
    bg = cv.morphologyEx(img, cv.MORPH_TOPHAT,
                        cv.getStructuringElement(cv.MORPH_ELLIPSE,(31,31)))
    img_eq = exposure.equalize_adapthist(bg, clip_limit=0.01)
    img_eq = filters.median(img_eq)

    # 2a. Multi-scale LoG
    sigmas   = np.arange(0.5, 1.5, 0.1)
    log_resp = np.zeros_like(img_eq)
    for s in sigmas:
        log_resp = np.maximum(log_resp,
                            s**2 * np.abs(filters.laplace(filters.gaussian(img_eq, s))))

    #loc th
    combo = log_resp > np.percentile(log_resp, 85)

    th      = filters.threshold_sauvola(img_eq, window_size=11, k=0.12)
    mask_in = np.logical_and(combo, img_eq > th)

    #morph
    mask = morphology.remove_small_objects(mask_in, 15)       # area in px²
    mask = morphology.remove_small_holes(mask, 20)
    mask = morphology.binary_opening(mask, morphology.disk(1))

    return mask.astype(np.uint8)





def multi_stage(stack, print_per_layer=False):
    output_data = np.zeros_like(stack, dtype=np.float32)
    original_stack = stack.copy()

    # Orientation 0:
    for idx in range(original_stack.shape[0]):
        processed_layer = process_layer(original_stack[idx])
        output_data[idx, :, :] += processed_layer

        num_labels, _, _, _ = cv.connectedComponentsWithStats(processed_layer, connectivity=8)
        if print_per_layer:
            print(f"Orientation 0, slice {idx}: {num_labels - 1} components detected")

    # # Orientation 1:
    # stack_o1 = np.transpose(original_stack, (1, 0, 2))
    # for idx in range(stack_o1.shape[0]):
    #     processed_layer = process_layer(stack_o1[idx])

    #     output_data[:, idx, :] += processed_layer

    #     num_labels, _, _, _ = cv.connectedComponentsWithStats(processed_layer, connectivity=8)
    #     if print_per_layer:
    #         print(f"Orientation 1, slice {idx}: {num_labels - 1} components detected")

    # # Orientation 2:
    # stack_o2 = np.transpose(original_stack, (2, 1, 0))
    # for idx in range(stack_o2.shape[0]):
    #     processed_layer = process_layer(stack_o2[idx])

    #     processed_layer = np.transpose(processed_layer)
    #     output_data[:, :, idx] += processed_layer

    #     num_labels, _, _, _ = cv.connectedComponentsWithStats(processed_layer, connectivity=8)
    #     if print_per_layer:
    #         print(f"Orientation 2, slice {idx}: {num_labels - 1} components detected")
    
    binary_output = output_data >= 1

    #opening
    e_structure = np.ones((3, 3, 3))
    d_structure = np.ones((3, 3, 3))
    opened_data = scipy.ndimage.binary_erosion(binary_output, structure=e_structure)
    opened_data = scipy.ndimage.binary_dilation(opened_data, structure=d_structure)
    
    return opened_data



def overlay_tiff_stacks(stack1_dir, stack2_dir, output_dir, alpha=0.5):
    os.makedirs(output_dir, exist_ok=True)

    stack1_files = natsort.natsorted(os.listdir(stack1_dir))
    stack2_files = natsort.natsorted(os.listdir(stack2_dir))

    if len(stack1_files) != len(stack2_files):
        raise ValueError("The input TIFF stacks must have the same number of frames.")

    for i, (file1, file2) in enumerate(zip(stack1_files, stack2_files)):
        img1 = tiff.imread(os.path.join(stack1_dir, file1))
        img2 = tiff.imread(os.path.join(stack2_dir, file2))

        if img1.shape != img2.shape:
            raise ValueError("All corresponding images must have the same dimensions.")

        blended_img = (img1 + 0.1 * img2).astype(np.uint16)
        tiff.imwrite(os.path.join(output_dir, f"frame_{i:04d}.tif"), blended_img)

    print(f"Overlay saved in {output_dir}")

def connected_components_3d(volume, connectivity=26):
    structure = np.ones((3, 3, 3)) if connectivity == 26 else None
    labeled_volume, num_features = scipy.ndimage.label(volume, structure=structure)

    return labeled_volume, num_features

if __name__ == '__main__':

    images = load_data(foldername)
    

    filtered_images = multi_stage(images, print_per_layer = True)

    labeled, num_features = connected_components_3d(filtered_images)

    print(f"# nuclei: {num_features}")

    print(f"density: {num_features/images.size}")
    

    output_folder = "multi_stage"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    for i, layer in enumerate(filtered_images):
        tiff.imwrite(os.path.join(output_folder, f"{i}.tiff"), layer)

    overlay_tiff_stacks('multi_stage', foldername, "superimposed_multi_stage", alpha=0.5)
    dc.overlay_contours_3d('multi_stage', foldername, "contours", alpha=0.3)