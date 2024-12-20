import logging
import os
import shutil
import subprocess
from typing import Optional, Tuple

from adam_core.orbits import Orbits

from adam_impact_study.conversions import fo_to_adam_orbit_cov

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_fo_od(
    fo_input_file: str,
    obj_id: str,
    FO_DIR: str,
    RESULT_DIR: str,
) -> Tuple[Orbits, Optional[str]]:
    """
    Run the find_orb orbit determination process for each object

    Parameters
    ----------
    fo_input_file : str
        Name of the find_orb input file.
    FO_DIR : str
        Directory path where the find_orb executable is located.
    RUN_DIR : str
        Directory path where the script is being run.
    RESULT_DIR : str
        Directory path where the results will be stored.

    Returns
    -------
    orbit : `~adam_core.orbits.orbits.Orbits`
        Orbit object containing the orbital elements and covariance matrix.
    error : str
        Error message if the orbit determination failed.
    """

    # Generate the find_orb commands
    fo_output_folder = os.path.join(RESULT_DIR, f"{obj_id}")
    os.makedirs(fo_output_folder, exist_ok=True)
    # fo_command = (
    #     f"{FO_DIR}/fo {fo_input_file} -O {fo_output_folder}" f" -D {FO_DIR}/environ.dat"
    # )
    RUN_DIR = "/Users/natetellis/code/adam_impacts_study/demo"
    fo_command = (
        f"cd {FO_DIR}; ./fo {fo_input_file} "
        f"-O {fo_output_folder}; cp -r {fo_output_folder} "
        f"{RESULT_DIR}/; cd {RUN_DIR}"
    )
    logger.info(f"Find Orb command: {fo_command}")

    # Ensure the output directory exists and copy the input file
    os.makedirs(f"{FO_DIR}/{fo_output_folder}", exist_ok=True)
    shutil.copyfile(
        f"{RESULT_DIR}/{fo_input_file}",
        f"{FO_DIR}/{fo_input_file}",
    )

    # Run find_orb and check for output
    subprocess.run(fo_command, shell=True)
    if not os.path.exists(f"{obj_id}/covar.json"):
        logger.info(f"No find_orb output for: {fo_output_folder}")
        return (Orbits.empty(), "No find_orb output")

    # Convert to ADAM Orbit objects
    orbit = fo_to_adam_orbit_cov(f"{fo_output_folder}")

    return (orbit, None)
