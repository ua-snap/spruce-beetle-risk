"""Setup file for pipeline notebook"""

import os
import shutil
import subprocess
from itertools import product
from pathlib import Path
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray
from wrf import PolarStereographic
from pyproj import Proj, Transformer
# project scripts
import slurm
from classify_clip_mask import run_classify_clip_mask


ncar_dir = Path(os.getenv("AK_NCAR_DIR"))
base_dir = Path(os.getenv("BASE_DIR"))
output_dir = Path(os.getenv("OUTPUT_DIR"))
scratch_dir = Path(os.getenv("SCRATCH_DIR"))
conda_init_script = Path(os.getenv("CONDA_INIT"))
slurm_email = Path(os.getenv("SLURM_EMAIL"))
partition = Path(os.getenv("SLURM_PARTITION"))
project_dir = Path(os.getenv("PROJECT_DIR"))
# binary directory from the current conda environment is appended to PATH
path_str = os.getenv("PATH")
# can use this to activate the anaconda-project env
ap_env = Path(path_str.split(":")[0]).parent

# met_dir = Path("/Data/Base_Data/Climate/AK_NCAR_12km/met")
# path to the input meteorological dataset
met_dir = ncar_dir.joinpath("met")

# path to directory where risk components datasets will be written
risk_comp_dir = scratch_dir.joinpath("risk_components")
risk_comp_dir.mkdir(exist_ok=True)

# directory where yearly risk datasets will be written
yearly_risk_dir = scratch_dir.joinpath("yearly_risk")
yearly_risk_dir.mkdir(exist_ok=True)

# directory where risk class dataset will be written
risk_class_dir = scratch_dir.joinpath("risk_class")
risk_class_dir.mkdir(exist_ok=True)

# output direcotry for risk class
out_risk_dir = output_dir.joinpath("risk_class")
out_risk_dir.mkdir(exist_ok=True)

# daymet_comp_fp = scratch_dir.joinpath("yearly_risk_components_daymet.nc")
# path to directory where slurm scripts (jobs and outputs) will be written
slurm_dir = scratch_dir.joinpath("slurm")
slurm_dir.mkdir(exist_ok=True, parents=True)

# script for computing yearly risk for a subset of the data
risk_script = project_dir.joinpath("compute_yearly_risk.py")

# Forest mask of Alaska in EPSG:3338
forest_fp = base_dir.joinpath("ak_forest_mask.tif")

# template raster for writing projected slice of NCAR data
temp_ncar_fp = scratch_dir.joinpath("ncar_template_3338.tif")
# template raster that has been clipped (cropped) to extent of forest mask
temp_ncar_clip_fp = scratch_dir.joinpath("ncar_template_clipped_3338.tif")
# forest mask that has been regridded to NCAR grid
ncar_forest_fp = scratch_dir.joinpath("ak_forest_mask_ncar_3338.tif")

models = [
    "CCSM4",
    "GFDL-ESM2M",
    "HadGEM2-ES",
    "MRI-CGCM3",
]

scenarios = ["rcp45", "rcp85"]

eras = ["2010-2039", "2040-2069", "2060-2099"]

# all projections will have years 2010-2099
# need to start with 2008 as yearly risk calculation
#   requires risk components from two years prior
full_future_era = "2008-2099"
