#!/usr/bin/env python3

import numpy as np
from tifffile import imread, imwrite
import os
import rasterio



def gen_labels(data: np.ndarray) -> np.ndarray:
    """
    generates ndvi labels for an input numpy array
    where channels include red and nir bands at indices 2 and 3 respectively.
    """
    red_band = data[:, :, 2].astype(np.float32)
    nir_band = data[:, :, 3].astype(np.float32)
    
    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)  # Adding a small constant to avoid division by zero
    
    # Generate labels based on NDVI thresholds
    labels = np.zeros_like(ndvi, dtype=np.uint8)
    labels[ndvi <= 0] = 0  # water
    labels[(ndvi > 0.0) & (ndvi <= 0.2)]= 1  # Non-vegetated
    labels[(ndvi > 0.2) & (ndvi <= 0.5)] = 2  # Moderately vegetated
    labels[ndvi > 0.5] = 3  # Densely vegetated
    
    return labels


def save_labels(labels: np.ndarray, output_path: str):
    """
    saves the labels numpy array as a tiff file at the specified output path,
    and as a .npy file.
    """
    from tifffile import imwrite
    np.save(output_path.replace('.tif', '.npy'), labels)
    # for the imwrite function, to make the image visible we scale the labels to 0-255
    imwrite(output_path, (labels * 85).astype(np.uint8))  


def import_tifs(directory: str) -> list[np.ndarray]:
    """
    imports all 4-band tiff files from a given directory and
    returns them as a list of numpy arrays
    """

    tiff_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    data_arrays = []

    for tiff_file in tiff_files:
        with rasterio.open(os.path.join(directory, tiff_file)) as src:
            data = src.read()
            data_arrays.append(np.transpose(data, (1, 2, 0)))  # Transpose to (height, width, channels)

    return data_arrays

def main():
    input_directory = "./input_images/"  # Change this to your directory path
    output_directory = "./output_labels/"  # Change this to your desired output path

    data_array = import_tifs(input_directory)[0]

    labels = gen_labels(data_array)
    save_labels(labels, output_directory+"labels.tif")


if __name__ == "__main__":
    main()
