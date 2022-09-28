"""Script for computing yearly risk for a given model, scenario, and years."""

import argparse
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import xarray as xr


def univoltine(tmin, tmax):
    """The function used for calculating the summer risk component, the 'survival' associated with univoltinism.
    
    Args:
        tmin (numpy.ndarray): a 1-D array of minimum temperatures for a given year
        tmax (numpy.ndarray): a 1-D array of maximum temperatures for a given year
        
    Returns:
        the summer risk component as a float between 0 and 1

    Notes:
        hard-coded "magic" values below are determined from available literature on univoltinsim in these beetles, where a relationship between prevlanece of univoltinism and accumulated heat above 17 C and below between 40 and 90 days post-first-flight, which occurs at 16 C. 
    """
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
    
    # easy if tmin is ever above 17 - count entire day
    # need to remember indices of values so we can exclude from
    #  the hourly estimator below
    hot_idx = tmin > 17
    k += 24 * hot_idx.sum()
    # discard indices that counted for entire days above 17C
    tmax = tmax[~hot_idx]
    tmin = tmin[~hot_idx]
    
    # also can immediately discard days where tmax is > 17C
    cold_idx = tmax < 17
    # discard indices that counted for entire days above 17C
    tmax = tmax[~cold_idx]
    tmin = tmin[~cold_idx]
    
    # need special treatment for tmin == 17 as well, as it would
    #  require division by zero in our estimation algorithm below.
    #  just assume it is above 17 for 75% of the time, or 18 hrs
    equal_idx = tmin == 17
    k += 18 * equal_idx.sum()
    # discard indices that counted for days where tmin == 17C
    tmax = tmax[~equal_idx]
    tmin = tmin[~equal_idx]
    
    # finally, can implement our estimation for days where tmin <17 and tmax>17
    h_est = (24 * (tmax - 17) / ((tmax - 17) + (17 - tmin)))
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
    """Execute the fall risk components for an array of minimum (daily) temperature values.
    
    Args:
        arr (numpy.ndarray): a 1-D array of minimum temperatures for a given winter
        
    Returns:
        fall_survival (float): the fall risk component as a value between 0 and 1
    """
    try:
        idx = np.where(arr <= -12)[0][0]
    except IndexError:
        if np.isnan(arr[0]):
            return np.nan
        else:
            return 1.0

    # Literature suggests that an ~21-day window of cooling of ~1/2 degree C per day 
    #  is about as extreme as most beetles can handle. Beyond that, mortality starts
    #  to occur, proportional to the severity / shock of that cooling.
    window = arr[idx : idx + 21]
    
    # cooling cutoff values - the upper and lower thr have different slopes
    #  to account for seemingly non-linear cold-tolerance accumulation through time.
    upper_thr = np.arange(-12, -22.5, -0.5)
    lower_thr = np.arange(-21, -42, -1)
    fall_survival = round(1 - ((upper_thr - window) / (upper_thr - lower_thr)).max(), 2)
    fall_survival = np.clip(fall_survival, 0.01, 1)

    return fall_survival


def winter_survival(tmin, snow):
    """Map a supplied minimum temperature to percent survival based on snowpack
    
    Args:
        tmin (float): a single value to map to a linear gradient of survival based on snowpack
        snow (str): snowpack level that has an insulating effect, either 'low', 'med', or 'high'
       
    Returns:
        the winter risk component as a float between 0 and 1
    """
    # the values below correspond to a linear mapping of minimum temperature (between
    #  the two bounds given, to % survival such that the bounds correpsond to 0 and 100% 
    #  survival. So if e.g. tmin == -30 under low snow, that will correspond to 50% 
    #  survival. The tmin-to-survival rate / slope is constant at -5% per degree for 
    #  any amount of snow, but the intercept changes for each. 
    if snow == "low":
        # linear ramp from -20 (100%) to -40 (0%) for no snowpack
        winter_survival = 200 + 5 * tmin
    elif snow == "med":
        # linear ramp from -30 (100%) to -50 (0%) for medium snowpack
        winter_survival = 250 + 5 * tmin
    elif snow == "high":
        # linear ramp from -40 (100%) to -60 (0%) for heavy  snowpack
        winter_survival = 300 + 5 * tmin
    else:
        raise ValueError("snow parameter must be one of low, med, or high")
    winter_survival = np.clip(winter_survival, 1, 100)

    return np.round(winter_survival / 100, 2)


def generate_ncar_filepaths(met_dir, tmp_fn, years, model, scenario):
    """Generate a sequence of yearly ncar filepaths
    
    Args:
        met_dir (pathlib.PosixPath): path to directory containing met data
        tmp_fn (str): template filename string ready to be formatted according to pattern model, scenario, year (CMIP5) or model, year (daymet)
        years (list): years to make filepaths for
        model (str): model name as used in file paths
        scenario (str): scenario name as used in filepaths - set to None for daymet
            
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
        fps = [met_dir.joinpath(model, tmp_fn.format(year)) for year in years]

    return fps


def compute_risk(u_t1, u_t2, un_t2, x2_t1, x2_t2, x3_t1, x3_t2):
    """Main equation for computing risk for a given year based on supplied survival parameter values. Written using symbology consistent with specs doc
    
    Args:
        u_t1 (float): univoltinsm survival for year t-1
        u_t2 (float): univoltinsm survival for year t-2
        un_t2 (float): univoltinsm survival for year t-2
        x2_t1 (float): fall survival for year t-1
        x2_t2 (float): fall survival for year t-2
        x3_t1 (float): winter survival for year t-1
        x3_t2 (float): winter survival for year t-2
        
    Returns:
        2D numpy.ndarray of risk
    """
    # survival based on univoltine predation rate
    p = 0.68
    # semivoltine predation survival was determined to be about 9 times lower than univoltine. 
    #  Overwinterin as adults is a gamechanger. 
    sv_p = 0.68 / 9
    # full, unsimplified equation
    # risk = (un_t2 * sv_p * x2_t2 * x2_t1 * x3_t2 * x3_t1) + ((u_t2 * p * x2_t2 * x3_t2) * (u_t1 * p * x2_t1 * x3_t1)) + (u_t2 * p * x2_t2 * x2_t1 * x3_t2 * x3_t1)
    # simplified by pulling out (x2_t2 * x2_t1 * x3_t2 * x3_t1)
    risk = ((un_t2 * sv_p) + (u_t2 * u_t1 * p ** 2) + (u_t2 * p)) * (
        x2_t2 * x2_t1 * x3_t2 * x3_t1
    )
    
    return risk


def process_risk_components(met_dir, tmp_fn, era, model, scenario=None):
    """Process the risk components for each year from climate data for a given model, scenario, and era.
    
    Args:
        met_dir (pathlib.PosixPath): path to the directory containing met data organized as folders named by model
        tmp_fn (str): template filename string
        era (str): era to be processed, of the form <start year>-<end year>
        model (str): model to be processed
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
        """Helper function to be used for the preprocess argument of xarray.open_mfdataset. The NCAR daymet files do not natively represent those as coordinate variables like the CMIP5 data do, so this function will just ensure that happens.
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
            # need to flip lat/lon arrays as well, since the values are flipped above
            "longitude": (["y", "x"], np.flipud(ds["longitude"].values)),
            "latitude": (["y", "x"], np.flipud(ds["latitude"].values)),
            "snow": (["snow"], snow_values),
        },
        # need to expand dims to add an extra for each for the single model, scenario
        #  we are working with.
        data_vars={
            "summer_survival": (
                ["year", "y", "x"],
                summer_survival_arr
            ),
            "fall_survival": (
                ["year", "y", "x"],
                fall_survival_arr,
            ),
            "winter_survival": (
                ["snow", "year", "y", "x"],
                winter_survival_arr
            ),
        },
        attrs=dict(description="Climate-based beetle risk components",),
    )

    return risk_comp_ds


def process_yearly_risk(risk_comp_ds):
    """Create yearly risk dataset from risk component dataset
    
    Args:
        risk_comp_ds (xarray.Dataset): dataset of risk components created using the process_risk_components function
        
    Returns:
        yearly_risk_ds (xarray.Dataset): dataset of yearly risk
    """
    # const_args = {"model": model, "scenario": scenario}
    snow_values = ["low", "med", "high"]
    # can only compute risk for years for which there are two previous
    #   years of risk components available
    years = risk_comp_ds.year.values[2:]
    yearly_risk_arrs = []
    for year in years:
        u_t2 = risk_comp_ds["summer_survival"].sel(year=(year - 2)).values
        u_t1 = risk_comp_ds["summer_survival"].sel(year=(year - 1)).values
        # "not univoltine"
        un_t2 = np.round(1 - u_t2, 2)
        x2_t2 = risk_comp_ds["fall_survival"].sel(year=(year - 2)).values
        x2_t1 = risk_comp_ds["fall_survival"].sel(year=(year - 1)).values

        year_snow_risk = []
        for snow in snow_values:
            x3_t2 = risk_comp_ds["winter_survival"].sel(year=(year - 2), snow=snow).values
            x3_t1 = risk_comp_ds["winter_survival"].sel(year=(year - 1), snow=snow).values

            year_snow_risk.append(compute_risk(u_t1, u_t2, un_t2, x2_t1, x2_t2, x3_t1, x3_t2))

        yearly_risk_arrs.append(np.array(year_snow_risk))

    yearly_risk_arr = np.swapaxes(np.array(yearly_risk_arrs), 0, 1)
    
    yearly_risk_ds = xr.Dataset(
        # need to expand dims to add an extra for each of model, scenario
        data_vars={"risk": (["snow", "year", "y", "x"], yearly_risk_arr)},
        coords={
            "year": (["year"], years),
            "longitude": (["y", "x"], risk_comp_ds["longitude"].values),
            "latitude": (["y", "x"], risk_comp_ds["latitude"].values),
            "snow": (["snow"], snow_values),
        },
        attrs=dict(description="Climate-based beetle risk",),
    )
    
    return yearly_risk_ds


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
        "--risk_comp_fp",
        help="File path to write risk component dataset",
    )
    parser.add_argument(
        "--yearly_risk_fp",
        help="File path to write yearly risk dataset",
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

    # parse the args and unpack
    args = parser.parse_args()
    met_dir = Path(args.met_dir)
    
    # process risk components
    print("Creating yearly risk components from input data")
    tic = time.perf_counter()
    
    risk_comp_ds = process_risk_components(
        met_dir,
        args.tmp_fn,
        args.era,
        args.model,
        args.scenario,
    )

    risk_comp_ds.to_netcdf(args.risk_comp_fp)
    print((
        "Yearly risk component dataset created in "
        f"{round((time.perf_counter() - tic) / 60, 1)}m, "
        f"written to {args.risk_comp_fp}"
    ))
    
    # process yearly risk
    print("Creating yearly risk dataset from risk components")
    tic = time.perf_counter()
    
    yearly_risk_ds = process_yearly_risk(risk_comp_ds)
    yearly_risk_ds.to_netcdf(args.yearly_risk_fp)
    print((
        f"Yearly risk dataset created in {round(time.perf_counter() - tic)}s, "
        f"written to {args.yearly_risk_fp}"
    ))
