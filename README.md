# Climate-driven spruce beetle risk for Alaska

This repo contains the code used for producing a dataset of projected climate-driven risk of spruce beetle outbreak for forested areas of Alaska for the 21st century. It contains a python implementation of a model consisting of climate-driven factors that are important to survival and reproduction of the spruce beetle that has been responsible for major pockets of spruce mortality in Alaska. The pipeline herein makes use of daily CMIP5 temperature projections to model estimates of outbreak risk over future 30-year eras. 

## Running the pipeline

The pipeline can be exectued using the [pipeline](pipeline.ipynb) notebook.

This repo utilizes `anaconda-project` for dependency management and pipeline execution. With an updated version of `anaconda-project` installed and available in a shell, run 

```
anaconda-project run pipeline
```

to create the necessary environment and open the pipeline notebook. 
Executing this command for the first time may take a while (~15 minutes or more). 

#### Environment variables

Running the above command will also ensure that the required environment variables are set, which are:

`AK_NCAR_DIR`

The directory containing the 12km Alaska.
* default value: `/Data/Base_Data/Climate/AK_NCAR_12km`

`BASE_DIR`

The base directory for storing project data that should be backed up.
* default value: `/workspace/Shared/Tech_Projects/beetles/project_data`

`OUTPUT_DIR`

The output directory where final products are placed.
* default value: `/workspace/Shared/Tech_Projects/beetles/final_products`

`SCRATCH_DIR`

The scratch directory for storing project data which does not need to be backed up. This one does not have a default and can be set to a directory in your personal scratch space in `/atlas_scratch` if working on Atlas.

## Model info 

This section goes into a little bit of detail on the model / algorithm itself. 

(needs work)
* The final categorical risk value for a summary era / time period requires yearly risk values which are in (1, 0] and typically on the order of 0.1 - 0.001.
* Each yearly risk value is based on three components for each year: a univoltine component, a fall cooling survival component, and a winter cold survival component
* Risk components can be defined for each year, but the yearly risk value requires some risk components that go back as far as two years. 
* The final formula for yearly risk is:
