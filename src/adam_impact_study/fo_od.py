import logging
import os
import pathlib
import shutil
import subprocess
from typing import Optional, Tuple

from adam_core.observations.ades import ADESObservations
from adam_core.orbits import Orbits

from adam_impact_study.conversions import (
    fo_to_adam_orbit_cov,
    rejected_observations_from_fo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LINUX_JPL_PATH = pathlib.Path(__file__).parent.parent.parent / "find_orb/linux_p1550p2650.430t"


def _create_fo_working_directory(FO_DIR: str, working_dir: str) -> str:
    os.makedirs(working_dir, exist_ok=True)
    # List of required files to copy from FO_DIR to current directory
    required_files = [
        "ObsCodes.htm",
        "jpl_eph.txt",
        "orbitdef.sof",
        "rovers.txt",
        "xdesig.txt",
        "cospar.txt",
        "efindorb.txt",
        "odd_name.txt",
        "sigma.txt",
    ]

    # Copy required files to the current directory
    for file in required_files:
        src = os.path.join(FO_DIR, file)
        dst = os.path.join(working_dir, file)
        shutil.copy2(src, dst)  # Copy to current directory

    # Template in the JPL path to environ.dat
    environ_dat_template = pathlib.Path(__file__).parent / "environ.dat.tpl"
    with open(environ_dat_template, "r") as file:
        environ_dat_content = file.read()
    environ_dat_content = environ_dat_content.format(LINUX_JPL_FILENAME=LINUX_JPL_PATH.absolute())
    with open(os.path.join(working_dir, "environ.dat"), "w") as file:
        file.write(environ_dat_content)

    return working_dir


def run_fo_od(
    FO_DIR: str,
    RESULT_DIR: str,
    fo_input_file: str,
    run_name: str,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """
    Run the find_orb orbit determination process for each object

    Parameters
    ----------
    fo_input_file : str
        Name of the find_orb input file.
    run_name : str
        Unique identifier for the fo run
    FO_DIR : str
        Directory path where the find_orb executable is located.
    RESULT_DIR : str
        Directory path where the results will be stored.

    Returns
    -------
    orbit : `~adam_core.orbits.orbits.Orbits`
        Orbit object containing the orbital elements and covariance matrix.
    rejected_observations : `~adam_core.observations.ades.ADESObservations`
        Rejected observations from the orbit determination.
    error : str
        Error message if the orbit determination failed.
    """

    # We create a working directory, as fo generates files in the same
    # directory as configuration files and we don't want our jobs to overwrite
    # each other.
    working_dir = os.path.join(RESULT_DIR, f"{run_name}")
    working_dir = _create_fo_working_directory(FO_DIR, working_dir)

    fo_command = (
        f"{FO_DIR}/fo {fo_input_file} "
        f"-c " # Combine all observations as if you only had one object
        f"-D {working_dir}/environ.dat"
    )

    logger.info(f"Find Orb command: {fo_command}")
    # Run find_orb and capture output
    output = subprocess.run(
        fo_command,
        shell=True,
        cwd=working_dir,
        text=True,  # Convert output to string
        capture_output=True,  # Capture stdout and stderr
    )
    
    # Log the output during debugging
    logger.debug(f"{output.stdout}")
    
    if output.returncode != 0:
        logger.info(f"Find_orb failed with return code: {output.returncode}")
        logger.info(f"Error output: {output.stderr}")  # Log error output on failure
        return (Orbits.empty(), ADESObservations.empty(), "Find_orb failed")
    if not os.path.exists(f"{working_dir}/covar.json"):
        logger.info(f"No find_orb output for: {working_dir}")
        return (Orbits.empty(), ADESObservations.empty(), "No find_orb output")

    # Convert to ADAM Orbit objects
    orbit = fo_to_adam_orbit_cov(f"{working_dir}")
    rejected_observations = rejected_observations_from_fo(f"{working_dir}")
    return (orbit, rejected_observations, None)
