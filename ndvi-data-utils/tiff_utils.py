import numpy as np
import tifffile
from pathlib import Path

def tiff_to_numpy_array(file_path: Path) -> np.ndarray:
    """
    Read a tiff from disk and convert it to a NumPy array. the array shape will be (height, width, bands).
    For our data, dtype is uint16.
    """

    array = tifffile.imread(file_path)

    return array

def numpy_array_to_tiff(array: np.ndarray, file_path: Path, dtype: str = 'uint16') -> None:
    """
    Save a NumPy array as a tiff file to disk.
    Assumes the array shape is (height, width, bands).

    dtype: data type to save the array as. default is 'uint16'. it really should stay this way because it is very annoying
    to debug data type issues with tiff files. you think a thing works, and it does,
    but you cant see the difference because 255 is basically black in uint16
    """

    if array.ndim != 3:
        raise ValueError("Input array must be 3D with shape (height, width, bands).")

    if dtype != "uint16":
        print("Warning: Saving tiff with dtype other than uint16 may lead to unexpected results.")

    tifffile.imwrite(file_path, array.astype(dtype))

