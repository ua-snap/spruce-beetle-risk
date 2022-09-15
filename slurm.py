"""Functions for orchestrating slurm jobs"""

import subprocess


def get_yearly_fps(slurm_dir, model, era, scenario=None):
    """Create the slurm file paths needed for the yearly risk processing
    
    Args:
        slurm_dir (pathlib.PosixPath): path to directory where slurm files should be written
        model (str): model to work on, whether one of the CMIP5 models or Daymet
        era (str): era to work on of the form YYYY-YYYY, <start year>-<end year>
        scenario (str): scenario to work on, defaults to None for daymet data
    
    Returns:
        tuple of (sbatch_fp, sbatch_out_fp), where sbatch_fp is the sbatch job file,
            and sbatch_out_fp is the output file
    """
    if scenario:
        attr_str = f"{model}_{scenario}_{era}"
    else:
        attr_str = f"{model}_{era}"
    sbatch_fp = slurm_dir.joinpath(f"yearly_risk_{attr_str}.slurm")
    sbatch_out_fp = slurm_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%j.out"))
    
    return sbatch_fp, sbatch_out_fp


def write_sbatch_yearly_risk(
    slurm_email,
    partition,
    conda_init_script,
    ap_env,
    sbatch_fp,
    sbatch_out_fp,
    risk_script,
    met_dir,
    tmp_fn,
    risk_comp_fp,
    yearly_risk_fp,
    era,
    model,
    scenario,
):
    """Write the sbatch job script for copmuting the yearly
    risk dataset for a given era, model, and scenario
    
    Args:
        slurm_email (str): email to use for sbatch job
        parition (str): slurm partition to use
        conda_init_script (str): path to script to be used for initializing conda cli
        ap_env (str): path to anaconda-project conda environment (the one for this project)
        sbatch_fp (str): path to write sbatch job script to
        sbatch_out_fp (str): path to write sbatch job output script
        risk_script (str): path to python script for computing yearly risk
        met_dir (str): path to meteorological subdataset of NCAR 12km AK hydrologic dataset
        tmp_fn (str): template filename with braces as placeholders datacube coordinates (year, model, scenario)
        risk_comp_fp (str): path to where risk components dataset should be written
        yearly_risk_fp (str): path to where yearly risk dataset should be written
        era (str): era in form "<start year (YYYY)>-<end year (YYYY)>"
        model (str): model name as used in NCAR filepaths
        scenario (str): scenario name as used in NCAR filepaths
    
    Returns:
        None; writes the sbatch script to path at sbatch_fp
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        f"source {conda_init_script}\n"
        f"conda activate {ap_env}\n"
    )

    pycommands = "\n"
    pycommands += (
        f"python {risk_script} "
        f"--met_dir {met_dir} "
        f"--tmp_fn {tmp_fn} "
        f"--risk_comp_fp {risk_comp_fp} "
        f"--yearly_risk_fp {yearly_risk_fp} "
        f"--era {era} "
        f"--model {model} "
        f"--scenario {scenario} "
    )
    commands = sbatch_head + pycommands

    with open(sbatch_fp, "w") as f:
        f.write(commands)


def submit_sbatch(sbatch_fp):
    """Submit a script to slurm via sbatch
    
    Args:
        sbatch_fp (pathlib.PosixPath): path to .slurm script to submit
        
    Returns:
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_fp)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id
