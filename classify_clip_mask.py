"""Some utilities for assisting with the data pipeline"""

import subprocess
import numpy as np
import rasterio as rio
import xarray as xr


def clip_with_gdal(src_fp, cut_fp):
    """Clip (crop) a file with gdal(warp)
    
    Args:
        src_fp (pathlib.PosixPath): path to file to be clipped 
        cut_fp (pathlike): path to shapefile to use for clipping
    
    Returns:
        clipped_fp (pathlib.PosixPath): path to the raster that resulted from clipping src_fp to cut_fp
    """
    clipped_fp = src_fp.parent.joinpath(src_fp.name.replace(".tif", "_clip.tif"))
    clipped_fp.unlink(missing_ok=True)
    _ = subprocess.call(
        [
            "gdalwarp",
            "-cutline",
            cut_fp,
            "-crop_to_cutline",
            "-q",
            src_fp,
            clipped_fp,
        ]
    )
    
    return clipped_fp


def classify_risk(arr):
    """Classify an array of risk values as being either low, medium, or high, encoded as 1, 2, or 3, respectively.
    
    Args:
        arr (numpy.ndarray): a 1-D array of risk values
        
    Returns:
        risk_class (int): risk class, either 1, 2, or 3
    """
    # this should only be used on 1-D arrays representing yearly
    #  risk at a single pixel
    assert(len(arr.shape) == 1)
    
    # risk classes are based on whether number of years over some
    #  threshold is greater than half of total years
    half = arr.shape[0] / 2
    
    # high and medium risk thresholds
    high_thr = 0.24
    med_thr = 0.12
    
    if any(np.isnan(arr)):
        risk_class = 0
    elif (arr >= high_thr).sum() >= half:
        risk_class = 3
    elif (arr >= med_thr).sum() >= half:
        risk_class = 2
    else:
        risk_class = 1
        
    return risk_class


def run_classify_clip_mask(yearly_risk_fp, era, snow, ncar_coords, wrf_proj_str, cut_fp, ncar_forest_fp, risk_class_fp):
    """Executes the process of classifying yearly risk to a requested summary period, reprojecting and clipping a classified 2-D slice to some extent, and masking with a forest raster.
    
    Args:
        yearly_risk_fp (pathlike): path tpo the yearly risk dataset corresponding to the supplied model and scenario
        era (str): era to be processed, of the form <start year>-<end year>
        snow (str): snowpack category to be processed, one of those present as coordinates in the yearly risk dataset
        ncar_coords (numpy.ndarray): dict of x and y coordinates for NCAR grid 
        wrf_proj_str (str): PROJ4 string for the WRF projection used for the NCAR grid
        cut_fp (pathlike): path to shapefile to be use for clipping risk data
        ncar_forest_fp (pathlike): path to the forest mask on the NCAR grid
        risk_class_fp (pathlib.PosixPath): path where the resulting risk class raster will bet written.
            
    Returns:
        None - writes to risk_class_fp and exits.
    """
    start_year, end_year = [int(year) for year in era.split("-")]
    year_sl = slice(start_year, end_year)
    with xr.open_dataset(yearly_risk_fp) as ds:
        # classify risk
        risk_class_arr = np.apply_along_axis(
            classify_risk,
            0,
            ds["risk"].sel(snow=snow, year=year_sl).values,
        ).astype(np.uint8)

    # create an xarray.DataArray for writing risk class
    #  data to georeferenced file
    da = xr.DataArray(
        data=risk_class_arr,
        dims=["y", "x"],
        coords=ncar_coords,
        attrs={"_FillValue": 0}
    )

    # write to a temporary file for clipping with gdal
    da.rio.write_crs(wrf_proj_str).rio.reproject("EPSG:3338").rio.to_raster(risk_class_fp)
    # run the clip
    class_clip_fp = clip_with_gdal(risk_class_fp, cut_fp)
    # rename the "clipped" file to remove the "_clip" substring  
    #   which overwrites risk_class_fp
    class_clip_fp.rename(risk_class_fp)
    # Then mask with forest raster
    with rio.open(risk_class_fp, "r+") as src:
        with rio.open(ncar_forest_fp) as mask_src:
            arr = src.read(1)
            mask = mask_src.read(1).astype(bool)
            arr[~mask] = 0
            src.write(arr, 1)
            
    return None
