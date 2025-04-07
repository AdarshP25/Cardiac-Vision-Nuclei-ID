import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
from numba import njit, prange
from math import sqrt
import scipy.ndimage
import natsort
import cv2 as cv
import multi_stage as ms

# Define folder names
foldername = "new_cropped"

def load_data(foldername):
    print("Reading .tiff files ...")
    tiff_files = sorted(
        [f for f in os.listdir(foldername) if f.endswith(".tiff") or f.endswith(".tif")]
    )
    for file in tiff_files:
        images = iio.imread(os.path.join(foldername, file))
        filtered_images = ms.multi_stage(images)
        labeled, num_features = connected_components_3d(filtered_images)
        print(f"{file}: {num_features} nuclei")
    return iio.imread(os.path.join(foldername, tiff_files[1]))


def connected_components_3d(volume, connectivity=26):
    structure = np.ones((3, 3, 3)) if connectivity == 26 else None
    labeled_volume, num_features = scipy.ndimage.label(volume, structure=structure)

    return labeled_volume, num_features


if __name__ == '__main__':

    load_data(foldername)
    




