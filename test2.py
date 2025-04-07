import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import natsort



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


overlay_tiff_stacks('top_hat_2', 'cropped_images', "superimposed_2", alpha=0.5)