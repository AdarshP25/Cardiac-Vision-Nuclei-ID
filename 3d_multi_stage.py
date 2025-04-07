import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

# Define folder names
foldername = "simple"

def differenceImage(img1, img2):
    a = img1 - img2
    b = np.uint8(img1 < img2) * 254 + 1
    return a * b

def load_data(foldername):
    print("Reading .tiff files ...")
    tiff_files = sorted(
        [f for f in os.listdir(foldername) if f.endswith(".tiff") or f.endswith(".tif")]
    )
    images = iio.imread(os.path.join(foldername, tiff_files[0]))
    return images

def three_dim_multi_stage(stack):
    # Denoise
    denoised = ndi.median_filter(stack, size=2)

    # DoG
    g_small = ndi.gaussian_filter(denoised, 1)
    g_large = ndi.gaussian_filter(denoised, 3)
    dog = differenceImage(g_small, g_large)

    # Normalize DoG
    dog = (dog - np.min(dog)) / (np.max(dog) - np.min(dog) + 1e-8) * 255

    # Adaptive Threshold
    thresh = (dog > threshold_local(dog, 21, 'gaussian')) * 255

    # Morphological Opening
    opened = ndi.grey_opening(thresh, size=3)

    # Connected Component Analysis
    min_size = 10
    max_size = 1000
    intensity_threshold = 4 

    # Label connected components
    labeled, num_features = ndi.label(opened)

    for label in range(1, num_features + 1):
        component_mask = (labeled == label)
        area = np.sum(component_mask)
        if area < min_size or area > max_size:
            opened[component_mask] = 0
        else:
            component_mean = np.mean(stack[component_mask])
            if component_mean < intensity_threshold:
                opened[component_mask] = 0

    return opened


def normalize_image(image):
    """ Normalize image to range [0, 255] for proper blending. """
    return ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255).astype(np.uint8)

if __name__ == '__main__':
    images = load_data(foldername)
    filtered_images = three_dim_multi_stage(images)

    # Normalize images for proper blending
    images_norm = normalize_image(images)
    filtered_norm = normalize_image(filtered_images)

    output_folder = "three_dim_multi_stage"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tiff.imwrite(os.path.join(output_folder, "filtered.tiff"), filtered_images)

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
