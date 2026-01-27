import os
import numpy as np
from make_labels import *
from tiff_utils import *


TIFF_DIR = "sample_data"
TIFF_NAME = "100680_nw_tile_0_1.tif"

def main():
    # get the current working directory
    cwd = os.getcwd()
    
    # load a sample tiff file
    sample_array: np.ndarray = tiff_to_numpy_array(os.path.join(cwd, TIFF_DIR, TIFF_NAME))
    
    blurred_array = blur_array(sample_array, kernel_size=5)
    
    # Calculate NDVI
    ndvi_array = calculate_ndvi(blurred_array)


    # Define NDVI thresholds for labeling
    ndvi_thresholds = [
        (0, -1.0, 0.0),    # Non-vegetation
        (85, 0.0, 0.2),     # Sparse vegetation
        (170, 0.2, 0.5),     # Moderate vegetation
        (255, 0.5, 1.0)      # Dense vegetation
    ]

    # Label NDVI. dtype is uint8, shape is (height, width)
    labeled_ndvi = label_ndvi(ndvi_array, ndvi_thresholds)

    # convert labeled_ndvi to 3D array for saving as tiff
    labeled_ndvi = np.expand_dims(labeled_ndvi, axis=2)
    labeled_ndvi = np.repeat(labeled_ndvi, 3, axis=2)  # repeat to make 3 bands
    
    # scale to uint16 range
    labeled_ndvi = (labeled_ndvi.astype(np.uint16) * 257)
    # at this point, labeled_ndvi is dtype uint16, shape (height, width, 3)

    #write it to disk for visualization
    numpy_array_to_tiff(labeled_ndvi, os.path.join(cwd, TIFF_DIR, "labeled_ndvi_" + TIFF_NAME))

    # save the array to disk as a numpy file
    np.save(os.path.join(cwd, TIFF_DIR, "ndvi_array.npy"), ndvi_array)



if __name__ == "__main__":
    main()
