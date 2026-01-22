from osgeo import gdal
from osgeo.gdal import Dataset as GDALDataset
import numpy as np
from pathlib import Path

def load_geotiff_image(file_path: Path) -> GDALDataset:
    """
    Load a GeoTIFF image using GDAL and return the GDAL Dataset.
    """
    ds: GDALDataset|None = gdal.Open(file_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open the file: {file_path}")
    return ds

def get_shape_of_gdal_dataset(ds: GDALDataset) -> tuple[int, int, int]:
    """
    Get the shape of a GDAL Dataset as (bands, rows, cols).
    """
    bands = ds.RasterCount
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    return (bands, rows, cols)

def gdaldataset_to_numpy_array(ds: GDALDataset) -> np.ndarray:
    """
    Convert a GDAL Dataset to a NumPy array.
    """
    bands, rows, cols = get_shape_of_gdal_dataset(ds)

    array: np.ndarray = np.zeros((bands, rows, cols))

    for b in range(1, bands + 1):
        band = ds.GetRasterBand(b)
        array[b - 1, :, :] = band.ReadAsArray()

    return array

