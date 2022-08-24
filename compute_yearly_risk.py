"""Script for computing yearly risk for a given model, scenario, and years."""

import argparse
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import xarray as xr


def univoltine(tmin, tmax):
    try:
        idx = np.where(tmax >= 16)[0][0]
    except IndexError:
        if np.isnan(tmin[0]):
            return np.nan
        else:
            return 0

    tmin = tmin[idx + 40 : idx + 90]
    tmax = tmax[idx + 40 : idx + 90]
    # hour counter
    k = 0
    # easy if tmin ever above 17
    # need to remember indices of values so we can exclude from
    #  the hourly estimator
    hot_idx = tmin > 17
    k += 24 * hot_idx.sum()
    # discard indices that counted for entire days above 17C
    tmax = tmax[~hot_idx]
    tmin = tmin[~hot_idx]
    # need special treatment for tin == 17 as well, as it would
    #  require division by zero in our estimation algorithm next.
    #  just assume it is above 17 for 75% of the time, or 18 hrs
    equal_idx = tmin == 17
    k += 18 * equal_idx.sum()
    # discard indices that counted for days where tmin == 17C
    tmax = tmax[~equal_idx]
    tmin = tmin[~equal_idx]
    # then, multiply percent of temp difference above 17 by 24
    #  to get estimate of hours above 17
    # then get the estimate of remaining hours above 17 and add to
    #  running total
    h_est = ((tmax - 17) / (17 - tmin)) / 2 * 24
    h_est[h_est < 0] = 0
    k += h_est.sum()

    # then determine "survival" due to univoltinism
    if k < 40:
        u = 0
    elif 40 <= k < 225:
        u = (k - 40) / 7.4
    elif 225 <= k < 412:
        u = 25 + (k - 225) / 2.5
    else:
        u = 100

    return round(u / 100, 2)


def fall_survival(arr):
    """Execute the fall survival algorithm for an
    array of temperature minimums for a single year.
    """
    try:
        idx = np.where(arr <= -12)[0][0]
    except IndexError:
        if np.isnan(arr[0]):
            return np.nan
        else:
            return 1.0

    # return 0 if tmin is ever less than -30
    if arr.min() < -30:
        return 0.01

    window = arr[idx : idx + 21]
    # cooling cutoff values
    thr_arr = np.arange(-12, -22.5, -0.5)
    dd = thr_arr - window
    # count only positive values and sum
    dd = dd[dd > 0].sum()
    # ensure value is between 1 and 100
    fall_survival = np.clip(100 - (dd * 4.76), 1, 100)

    return round(fall_survival / 100, 2)


def winter_survival(tmin, snow):
    """Map a supplied minimum temperature to percent survival
    based on snowpack
    """
    if snow == "low":
        # linear ramp from -20 (100%) to -40 (0%) for no snowpack
        winter_survival = 200 + 5 * tmin
    elif snow == "med":
        # linear ramp from -30 (100%) to -50 (0%) for no snowpack
        winter_survival = 250 + 5 * tmin
    elif snow == "high":
        # linear ramp from -40 (100%) to -60 (0%) for no snowpack
        winter_survival = 300 + 5 * tmin
    else:
        raise ValueError("snow parameter must be one of low, med, or high")
    winter_survival = np.clip(winter_survival, 1, 100)

    return np.round(winter_survival / 100, 2)


def generate_ncar_filepaths(met_dir, tmp_fn, years, model, scenario):
    """Generate a sequence of yearly ncar filepaths
    
    Args:
        met_dir (pathlib.PosixPath): path to directory containing met data
        tmp_fn (str): template filename string ready to be formatted according
            to pattern model, scenario, year (CMIP5) or model, year (daymet)
        years (list): years to make filepaths for
        model (str): model nanme as used in file paths
        scenario (str): scenario name as used in filepaths - set to None for 
            daymet
            
    Returns:
        filepaths for requested NCAR data subset
    """
    if scenario:
        # CMIP5 data
        fps = [
            met_dir.joinpath(model, scenario, tmp_fn.format(model, scenario, year))
            for year in years
        ]
    else:
        # daymet files will not have a scenario
        fps = [met_dir.joinpath(model, tmp_fn.format(model, year)) for year in years]

    return fps


def read_xarray(fp):
    ds = xr.load_dataset(fp)
    return ds


def compute_risk(u_t1, u_t2, un_t2, x2_t1, x2_t2, x3_t1, x3_t2):
    """Main equation for computing risk for a given year based on
    supplied survival parameter values. Written using symbology 
    consistent with specs doc
    
    Args:
        u_t1 (float): univoltinsm survival for year t-1
        u_t2 (float): univoltinsm survival for year t-2
        un_t2 (float): univoltinsm survival for year t-2
        x2_t1 (float): fall survival for year t-1
        x2_t2 (float): fall survival for year t-2
        x3_t1 (float): winter survival for year t-1
        x3_t1 (float): winter survival for year t-2
        
    Returns:
        2D numpy.ndarray of risk
    """
    # univoltine predation
    p = 0.68
    # semioltine predation
    sv_p = 0.68 / 9
    risk = ((un_t2 * sv_p) + (u_t2 * u_t1 * p ** 2) + (u_t2 * p)) * (
        x2_t2 * x2_t1 * x3_t2 * x3_t1
    )
    return risk


def process_risk_components(met_dir, tmp_fn, era, model, ncpus, scenario=None):
    """Process the risk components for each year from climate data
    for a given model, scenario, and era.
    
    Args:
        met_dir (pathlib.PosixPath): path to the directory containing met data organized as
            folders named by model
        tmp_fn (str): template filename string
        era (str): era to be processed, of the form <start year>-<end year>
        model (str): model to be processed
        ncpus (int): number of cpus to use for multiprocessing
        scenario (str): scenario to be processed (use None for daymet)
    
    Returns:
        risk_da (xarray.DataArray): DataArray of risk with dimensions model, scenario,
            snow load level, year, y index, x index
    """
    yearly_risk_arrs = []
    start_year, end_year = era.split("-")
    start_year = int(start_year)
    end_year = int(end_year)
    years = np.arange(start_year, end_year + 1)

    fps = generate_ncar_filepaths(met_dir, tmp_fn, years, model, scenario)

    def force_latlon_coords(ds):
        """Helper function to be used for the preprocess argument of xarray.open_mfdataset.
        The NCAR daymet files do not natively represent those as coordinate variables like
        the CMIP5 data do, so this function will just ensure that happens.
        """
        return ds.assign_coords(
            {coord: ds[coord] for coord in ["latitude", "longitude"]}
        )

    fall_survival_list = []
    winter_survival_list = []
    summer_survival_list = []

    # Pool-ing seemed to be faster than using this using open_mfdataset for a single job/node, but when
    #  submitted altogether things were not completing in reasonable time. So, going with
    #  this for now.
    with xr.open_mfdataset(fps, preprocess=force_latlon_coords) as ds:
        for year in years:
            winter_tmin = (
                ds["tmin"].sel(time=slice(f"{year - 1}-07-01", f"{year}-06-30")).values
            )
            tmax = ds["tmax"].sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values
            tmin = ds["tmin"].sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values

            fall_survival_list.append(
                np.apply_along_axis(fall_survival, 0, winter_tmin)
            )
            # need to iterate over axes indices for summer "survival" because
            #  both tmin and tmax arrays are needed
            summer_survival_arr = np.empty(tmin.shape[1:])
            for i, j in product(range(tmin.shape[1]), range(tmin.shape[2])):
                summer_survival_arr[i, j] = univoltine(tmin[:, i, j], tmax[:, i, j])
            summer_survival_list.append(summer_survival_arr)

            # each year will have three risk arrays, one for each level of snowpack
            winter_snow_list = []
            snow_values = ["low", "med", "high"]
            for snow in snow_values:
                winter_snow_list.append(winter_survival(winter_tmin.min(axis=0), snow))
            winter_survival_list.append(np.array(winter_snow_list))

    # make into arrays
    # flip along y axis to orient properly
    summer_survival_arr = np.flip(np.array(summer_survival_list), 1)
    fall_survival_arr = np.flip(np.array(fall_survival_list), 1)
    # swap the year and snow axes for more intuitive structure
    #  to (snow, year, y, x) from (year, snow, y, x)
    winter_survival_arr = np.flip(np.swapaxes(np.array(winter_survival_list), 0, 1), 2)
    risk_comp_ds = xr.Dataset(
        coords={
            "year": (["year"], years),
            "model": (["model"], [model]),
            "scenario": (["scenario"], [scenario]),
            # need to flip lat/lon arrays as well, since the values are flipped above
            "longitude": (["y", "x"], np.flipud(ds["longitude"].values)),
            "latitude": (["y", "x"], np.flipud(ds["latitude"].values)),
            "snow": (["snow"], snow_values),
        },
        # need to expand dims to add an extra for each for the single model, scenario
        #  we are working with
        data_vars={
            "summer_survival": (
                ["model", "scenario", "year", "y", "x"],
                np.expand_dims(summer_survival_arr, (0, 1)),
            ),
            "fall_survival": (
                ["model", "scenario", "year", "y", "x"],
                np.expand_dims(fall_survival_arr, (0, 1)),
            ),
            "winter_survival": (
                ["model", "scenario", "snow", "year", "y", "x"],
                np.expand_dims(winter_survival_arr, (0, 1)),
            ),
        },
        attrs=dict(description="Climate-based beetle risk",),
    )

    return risk_comp_ds


if __name__ == "__main__":
    # parse some args
    parser = argparse.ArgumentParser(
        description="Compute yearly risk for a given model, scenario, and years."
    )
    parser.add_argument(
        "--met_dir", help="Path to directory containing NCAR AK dataset met files"
    )
    parser.add_argument(
        "--tmp_fn",
        help="Template filename string with gaps for model, scenario, and year",
    )
    parser.add_argument(
        "--era",
        help="Era, or range of years to process given as '<start year>-<end year>",
    )
    parser.add_argument(
        "--model", help="Model name, as given in dataset file paths",
    )
    parser.add_argument(
        "--scenario", help="Scenario name, as given in dataset file paths",
    )
    parser.add_argument(
        "--ncpus", type=int, help="Number of CPUs to use for multiprocessing",
    )
    parser.add_argument(
        "--risk_fp", help="Number of CPUs to use for multiprocessing",
    )

    # parse the args and unpack
    args = parser.parse_args()
    met_dir = Path(args.met_dir)

    tic = time.perf_counter()
    # create yearly risk dataarray
    risk_da = process_risk_array(
        met_dir, args.tmp_fn, args.era, args.model, args.scenario, args.ncpus
    )
    print(f"Yearly risk array created, {round((time.perf_counter() - tic) / 60, 1)}m")

    tic = time.perf_counter()
    # save the risk dataarray in scratch space
    risk_da.to_netcdf(args.risk_fp)
    print(
        f"Yearly risk array written to {args.risk_fp}, {round(time.perf_counter() - tic)}s"
    )
