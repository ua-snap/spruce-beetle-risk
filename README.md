# Climate-driven spruce beetle risk for Alaska

This repo contains the code used for producing a dataset of projected climate-driven risk of spruce beetle outbreak for forested areas of Alaska for the 21st century. It contains a python implementation of a model consisting of climatic factors that are important to survival and reproduction of the spruce beetle that has been responsible for major pockets of spruce mortality in Alaska. The pipeline herein makes use of daily CMIP5 temperature projections to model estimates of outbreak risk over future 30-year eras. 

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

`CONDA_INIT`

This should be a shell script for initializing conda in a blank shell that does not read the typical `.bashrc`, as is the case with new slurm jobs.

It should look like this, with the variable `CONDA_PATH` below being the path to parent folder of your conda installation, e.g. `/home/UA/kmredilla/miniconda3`:

```
__conda_setup="$('$CONDA_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_PATH/bin:$PATH"
    fi
fi
unset __conda_setup
```

`SLRUM_EMAIL`

The email address to use for slurm jobs.

`SLURM_PARTITION`

The name of parition to use for slurm jobs.


## Model info 

This section goes into a little bit of detail on the model / algorithm itself. 

* The final categorical risk value for a summary era / time period requires yearly risk values which are in [0, 1.1424] and typically on the order of 0.1 - 0.001. Risk is classified as follows:
    * **high**: risk >= 0.24 for half of years in era or more
    * **medium**: 0.12 >= risk > 0.24 for half of years in era or more (if not high)
    * **low**: not medium or high risk
* Each yearly risk value is based on three components for each year: a univoltine / summer component, a fall cooling survival component, and a winter cold survival component
* Risk components are defined for each year, but the yearly risk value requires some risk components that go back as far as two years. 
* The final formula for risk for a given year $t$ is:

$$((1 - u)_{t - 2} * \frac{P}{9} * X2_{t - 2} * X2_{t - 1} * X3_{t - 2} * X3_{t - 1}) + (u_{t - 2} * P * X2_{t - 2} * X3_{t - 2} * u_{t - 1} * P * X2_{t - 1} * X3_{t - 1}) + (u_{t - 2} * P * X2_{t - 2} * X2_{t - 1} * X3_{t - 2} * X3_{t - 1})$$

where

* $u$ is the univoltine component  
* $X2$ is the fall cooling component
* $X3$ is the winter cold component
* $P$ is the probability of surviving predation
