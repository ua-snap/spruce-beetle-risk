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
