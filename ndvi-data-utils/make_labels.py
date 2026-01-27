import numpy as np

def calculate_ndvi(array: np.ndarray) -> np.ndarray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from a numpy array.
    Assumes that the dataset has at least two bands: Red and NIR.
    Typically, Red is band 3 and NIR is band 4 in a standard RGBNIR image.
    This assumes the input dataset follows that convention.
    returns a 2D numpy array representing the NDVI values of shape (height, width).

    the input array shape is (height, width, bands)
    the dtype of the input array is uint16
    the dtype of the output NDVI array is float32
    the output array shape is (height, width)
    """
    # Extract Red and NIR bands
    red_band = array[:, :, 2].astype(np.float32)  # Band 3
    nir_band = array[:, :, 3].astype(np.float32)  # Band 4

    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)  # Adding a small value to avoid division by zero

    return ndvi

def label_ndvi(ndvi_array: np.ndarray, thresholds: list[tuple[int, float, float]]) -> np.ndarray:
    """
    Label the NDVI array based on provided thresholds.
    thresholds: List of tuples in the form (label, min_value, max_value)
    returns a labeled array where each pixel is assigned a label based on the NDVI value.

    input array is dtype float32 with values in range [-1, 1]
    input array shape is (height, width)
    output array is dtype uint8 with labels assigned
    the output array shape is (height, width)
    """
    labeled_array = np.zeros_like(ndvi_array, dtype=np.uint8)

    for label, min_val, max_val in thresholds:
        mask = (ndvi_array >= min_val) & (ndvi_array < max_val)
        labeled_array[mask] = label

    return labeled_array



def blur_array(array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply gaussian blur to a numpy array representing a 4-band image. each band is blurred independently.
    """
    from scipy.ndimage import gaussian_filter

    if array.ndim != 3:
        raise ValueError("Input array must be 3D with shape (height, width, bands).")

    blurred_array = np.empty_like(array)
    for band in range(array.shape[2]):
        blurred_array[:, :, band] = gaussian_filter(array[:, :, band], sigma=kernel_size)

    return blurred_array
