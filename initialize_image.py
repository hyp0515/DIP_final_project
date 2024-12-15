import os
import numpy as np
from glob import glob
from PIL import Image

def preprocess_fingerprint_dataset(input_dir, output_fname, num_fingers=10, num_impressions=8, output_size=(160, 160)):
    """
    Preprocess all fingerprint .tif images and organize into (10, 8, 160, 160, 1).

    Args:
        input_dir (str): Path to the directory containing .tif files.
        num_fingers (int): Number of fingers in the dataset.
        num_impressions (int): Number of impressions per finger.
        output_size (tuple): Desired output size (height, width).

    Returns:
        numpy.ndarray: Dataset as a NumPy array of shape (10, 8, 160, 160, 1).
    """
    # Initialize an empty array for the dataset
    dataset = np.zeros((num_fingers, num_impressions, output_size[0], output_size[1], 1), dtype=np.float32)

    # Sort file paths to ensure consistent ordering
    file_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')])
    # Check if the total number of files matches expectations
    expected_files = num_fingers * num_impressions
    if len(file_paths) != expected_files:
        raise ValueError(f"Expected {expected_files} .tif files, but found {len(file_paths)}")

    # Preprocess each file
    for finger_idx in range(num_fingers):
        for impression_idx in range(num_impressions):
            # Calculate the file index
            file_index = finger_idx * num_impressions + impression_idx
            file_path = file_paths[file_index]
            
            # Load the image using Pillow
            img = Image.open(file_path)

            # Convert to grayscale (if not already)
            img = img.convert("L")
            
            # Resize the image to the desired size
            img_resized = img.resize(output_size)
            
            # Convert to NumPy array and normalize to [0, 1]
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Add a channel dimension and store in the dataset
            dataset[finger_idx, impression_idx, :, :, 0] = img_array

    np.save(output_fname+".npy", dataset)
    return