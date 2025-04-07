import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
from numba import njit, prange
from math import sqrt
import scipy.ndimage
import natsort

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



@njit(parallel=True)
def filter_tophat_2d(image, r_inner, r_outer, threshold):
    y_dim, x_dim = image.shape
    data = image.astype(np.float32)
    
    padded_shape = (y_dim + 2 * r_outer, x_dim + 2 * r_outer)
    padded_data = np.zeros(padded_shape, dtype=np.float32)
    for y in range(y_dim):
        for x in range(x_dim):
            padded_data[y + r_outer, x + r_outer] = data[y, x]
    
    output_data = np.zeros_like(data, dtype=np.float32)
    
    for y in prange(r_outer, r_outer + y_dim):
        for x in range(r_outer, r_outer + x_dim):
            inner_max = -1e10
            outer_max = -1e10
            
            for dy in range(-r_outer, r_outer + 1):
                for dx in range(-r_outer, r_outer + 1):
                    val = padded_data[y + dy, x + dx]
                    dist = sqrt(dx * dx + dy * dy)
                    if dist <= r_inner:
                        if val > inner_max:
                            inner_max = val
                    elif dist <= r_outer:
                        if val > outer_max:
                            outer_max = val
            
            if inner_max - outer_max > threshold:
                output_data[y - r_outer, x - r_outer] = 2**16
    
    return output_data.astype(np.uint16)

def overlay_tiff_images(stack1_dir, stack2_dir, output_dir, alpha=0.5):
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
    images = load_data("well_behaved")
    
    output_folder = "top_hat_2d"
    os.makedirs(output_folder, exist_ok=True)
    
    for i, img in enumerate(images):
        img_filtered = filter_tophat_2d(scipy.ndimage.gaussian_filter(img, sigma=1), r_inner=3, r_outer=6, threshold=100)
        tiff.imwrite(os.path.join(output_folder, f"{i:04d}.tiff"), img_filtered)
    
    overlay_tiff_images("top_hat_2d", "well_behaved", "superimposed_2d", alpha=0.5)
