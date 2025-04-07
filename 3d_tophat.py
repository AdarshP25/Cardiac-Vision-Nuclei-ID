import vtk
import imageio.v3 as iio
import numpy as np
#import cupy as cp
import os
import json
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.ndimage


foldername = "cropped_whole_stack"

foldername_snapshots = "snapshots/"


n_snapshot = 1

def load_data(foldername):
    print("reading .tiff files ...")
    tiff_files = sorted([f for f in os.listdir(foldername) if f.endswith(".tiff") or f.endswith(".tif")])
    images = []
    for file in tiff_files:
        image_path = os.path.join(foldername, file)
        img = iio.imread(image_path)
        images.append(img)
    image_stack = np.stack(images, axis=0)
    return image_stack

def filter_contrast(numpy_array, kernel_size = 20):
    print("filtering array using contrast-filter ...")
    array_size = (numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2])
    #data = np.random.rand(*array_size, dtype=np.float32)
    data = np.asarray(numpy_array, dtype=np.float32)
    half_kernel = kernel_size // 2
    padded_data = np.pad(data, pad_width=half_kernel, mode='constant', constant_values=0)
    output_data = np.empty_like(data)
    for z in range(half_kernel, array_size[0] + half_kernel):
        for y in range(half_kernel, array_size[1] + half_kernel):
            for x in range(half_kernel, array_size[2] + half_kernel):
                region = padded_data[z-half_kernel:z+half_kernel+1, y-half_kernel:y+half_kernel+1, x-half_kernel:x+half_kernel+1] # extract kernel
                region_min = np.min(region)
                region_max = np.max(region)
                center_value = padded_data[z, y, x]
                output_data[z-half_kernel, y-half_kernel, x-half_kernel] = (center_value - region_min) / max(region_max - region_min, 1)
    output_numpy_array = output_data
    return output_numpy_array

def filter_tophat(array, r_inner, r_outer, threshold):
    print("Filtering array using top-hat filter ...")
    array_size = array.shape
    data = np.asarray(array, dtype=np.float32)
    padded_data = np.pad(data, pad_width=r_outer, mode='constant', constant_values=0)
    output_data = np.zeros_like(data)

    z_dim, y_dim, x_dim = array_size

    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(-r_outer, r_outer+1),
        np.arange(-r_outer, r_outer+1),
        np.arange(-r_outer, r_outer+1),
        indexing='ij'
    )

    distance = np.sqrt(z_grid**2 + y_grid**2 + x_grid**2)
    outer_mask = (distance <= r_outer)
    inner_mask = (distance <= r_inner)
    outer_only_mask = outer_mask & ~inner_mask

    for z in range(r_outer, z_dim + r_outer):
        print(f"layer: {z}")
        for y in range(r_outer, y_dim + r_outer):
            for x in range(r_outer, x_dim + r_outer):
                region = padded_data[z-r_outer:z+r_outer+1, y-r_outer:y+r_outer+1, x-r_outer:x+r_outer+1]
                outer_max = np.max(region[outer_only_mask])
                inner_max = np.max(region[inner_mask])
                if inner_max - outer_max > threshold:
                    output_data[z-r_outer, y-r_outer, x-r_outer] = inner_max

    return output_data


images = load_data(foldername)


i = 0
for layer in filter_tophat(images, 3, 6, 100):
    tiff.imwrite(f"normalized/{i}.tiff", layer)
    i += 1



