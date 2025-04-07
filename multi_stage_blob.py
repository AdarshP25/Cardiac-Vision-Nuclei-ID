import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
import cv2 as cv
import natsort
import scipy.ndimage

# Define folder names
foldername = "test_cube"  # Folder with original TIFF files

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
    print(f"Loaded {len(images)} images with shape {images[0].shape}")
    return image_stack

def process_layer_blob(img):
    # Denoise the image
    denoised = cv.medianBlur(img, 3)
    
    # Preprocessing: Difference of Gaussians (DoG)
    g_small = cv.GaussianBlur(denoised, (3, 3), 0)
    g_large = cv.GaussianBlur(denoised, (11, 11), 0)
    dog = cv.subtract(g_small, g_large)
    dog = cv.normalize(dog, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # Setup blob detection parameters
    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10    # minimum blob area (adjust as needed)
    params.maxArea = 100   # maximum blob area (adjust as needed)
    params.filterByColor = True
    params.blobColor = 255  # detect bright blobs (change if blobs are dark)
    # Disable shape filters to allow irregular blobs
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv.SimpleBlobDetector_create(params)
    # For blob detection, normalize the original image
    keypoints = detector.detect(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8))
    
    # Create a binary mask: draw each detected blob as a filled white circle
    mask = np.zeros_like(img, dtype=np.uint8)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        cv.circle(mask, (x, y), radius, 255, thickness=-1)
    
    # Compute contours on the binary mask (for irregular shape analysis)
    contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    return mask, keypoints, contours

def multi_stage_blob(stack):
    output_data = np.zeros_like(stack, dtype=np.float32)
    for i in range(len(stack)):
        processed_mask, keypoints, contours = process_layer_blob(stack[i])
        output_data[i] = processed_mask
        # Use connectedComponentsWithStats to mimic original print output
        num_labels, _, _, _ = cv.connectedComponentsWithStats(processed_mask, connectivity=8)
        print(f"Layer {i}: {num_labels - 1} components detected; {len(contours)} contours found")
    return output_data

def overlay_tiff_stacks(stack1_dir, stack2_dir, output_dir, alpha=0.5):
    os.makedirs(output_dir, exist_ok=True)
    stack1_files = natsort.natsorted(os.listdir(stack1_dir))
    stack2_files = natsort.natsorted(os.listdir(stack2_dir))
    
    print(f"Found {len(stack1_files)} files in '{stack1_dir}'")
    print(f"Found {len(stack2_files)} files in '{stack2_dir}'")
    
    if len(stack1_files) != len(stack2_files):
        raise ValueError("The input TIFF stacks must have the same number of frames.")
    
    for i, (file1, file2) in enumerate(zip(stack1_files, stack2_files)):
        path1 = os.path.join(stack1_dir, file1)
        path2 = os.path.join(stack2_dir, file2)
        img1 = tiff.imread(path1)
        img2 = tiff.imread(path2)
        if img1.shape != img2.shape:
            raise ValueError(f"Dimension mismatch: {file1} and {file2}")
        # Blend images: original image + 0.1 * blob mask
        blended_img = (img1 + 0.1 * img2).astype(np.uint16)
        out_path = os.path.join(output_dir, f"frame_{i:04d}.tif")
        tiff.imwrite(out_path, blended_img)
        print(f"Saved overlay: {out_path}")
    
    print(f"Overlay saved in '{output_dir}'")

def connected_components_3d(volume, connectivity=26):
    structure = np.ones((3, 3, 3)) if connectivity == 26 else None
    labeled_volume, num_features = scipy.ndimage.label(volume, structure=structure)
    return labeled_volume, num_features

if __name__ == '__main__':
    # Load image stack from folder
    images = load_data(foldername)
    
    # Process the image stack using blob detection and contour extraction
    filtered_images = multi_stage_blob(images)
    
    # Perform 3D connected-components analysis on the binary volume
    labeled, num_features = connected_components_3d(filtered_images)
    print(f"# blobs (3D connected): {num_features}")
    
    # Save processed blob masks
    output_folder = "multi_stage_blob"
    os.makedirs(output_folder, exist_ok=True)
    for i, layer in enumerate(filtered_images):
        out_path = os.path.join(output_folder, f"{i}.tiff")
        tiff.imwrite(out_path, layer)
        print(f"Saved blob mask: {out_path}")
    
    # Create an overlay of the original images with the blob masks
    overlay_tiff_stacks('multi_stage_blob', foldername, "superimposed_blob", alpha=0.5)
