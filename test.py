#%%

import imageio.v3 as iio
import numpy as np
import os

# Path to the folder containing the TIFF images
folder_path = "508930_545120"

# Get a sorted list of all TIFF files in the folder
tiff_files = sorted(
    [f for f in os.listdir(folder_path) if f.endswith(".tiff") or f.endswith(".tif")]
)

# Read each image and stack them into a 3D numpy array
images = []
for file in tiff_files:
    image_path = os.path.join(folder_path, file)
    img = iio.imread(image_path)
    images.append(img)

# Convert the list of 2D arrays into a single 3D array
image_stack = np.stack(images, axis=0)

print("Shape of 3D array:", image_stack.shape)

#%%