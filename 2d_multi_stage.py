import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
from numba import njit, prange
from math import sqrt
import scipy.ndimage
import natsort
import cv2 as cv
import matplotlib.pyplot as plt


# Define folder names
foldername = "well_behaved"

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
    # Denoise
    denoised = cv.medianBlur(img, 3)

    #DoG
    g_small = cv.GaussianBlur(denoised, (3, 3), 0)
    g_large = cv.GaussianBlur(denoised, (15, 15), 0)
    dog = cv.subtract(g_small, g_large)

    # threshhold
    thresh = cv.adaptiveThreshold(
        dog.astype(np.uint8), 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY, 
        21, 2
    )

    # morph
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    # Connected-component analysis
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(opened, connectivity=8)

    min_size = 20
    max_size = 100
    for label_idx in range(1, num_labels):  # label 0 is the background
        area = stats[label_idx, cv.CC_STAT_AREA]
        if area < min_size or area > max_size:
            opened[labels == label_idx] = 0


    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(opened, connectivity=8)

    brightened = cv.convertScaleAbs(opened, alpha=2.0, beta=50)  # Increase contrast and brightness

    return brightened


def multi_stage(stack):
    output_data = np.zeros_like(stack, dtype=np.float32)
    for i in range(len(stack)):
        output_data[i] = process_layer(stack[i])
    return output_data


def overlay_tiff_stacks(stack1_dir, stack2_dir, output_dir, alpha=0.5):
    """Overlay two TIFF stacks (stored as a series of 2D images) with a specified transparency level (alpha)."""
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

        blended_img = (alpha * img1 + (1 - alpha) * img2).astype(np.uint16)
        tiff.imwrite(os.path.join(output_dir, f"frame_{i:04d}.tif"), blended_img)

    print(f"Overlay saved in {output_dir}")

if __name__ == '__main__':

    images = load_data(foldername)
    

    filtered_images = multi_stage(images)
    

    output_folder = "multi_stage"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    for i, layer in enumerate(filtered_images):
        tiff.imwrite(os.path.join(output_folder, f"{i}.tiff"), layer)

    overlay_tiff_stacks('multi_stage', 'well_behaved', "superimposed_multi_stage", alpha=0.5)

    # Blended image stack
    alpha = 0.5
    blended_images = (alpha * images_norm + (1 - alpha) * filtered_norm).astype(np.uint8)

    #visualization
    fig, ax = plt.subplots()
    slice_idx = [0]

    img_display = ax.imshow(blended_images[slice_idx[0]], cmap='gray', origin='upper', interpolation='nearest')
    ax.set_title(f"Slice {slice_idx[0] + 1}/{blended_images.shape[0]}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Scroll event
    def on_scroll(event):
        if event.step < 0:  # Scroll down
            slice_idx[0] = min(slice_idx[0] + 1, blended_images.shape[0] - 1)
        elif event.step > 0:  # Scroll up
            slice_idx[0] = max(slice_idx[0] - 1, 0)
        img_display.set_data(blended_images[slice_idx[0]])
        ax.set_title(f"Slice {slice_idx[0] + 1}/{blended_images.shape[0]}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()
