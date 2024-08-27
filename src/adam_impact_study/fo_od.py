import json
import os
import shutil
import subprocess

import quivr as qv

from adam_impact_study.conversions import fo_to_adam_orbit_cov


def run_fo_od(
    fo_input_file,
    fo_output_folder,
    FO_DIR,
    RUN_DIR,
    RESULT_DIR,
):
    """
    Run the find_orb orbit determination process for each object

    Parameters
    ----------
    fo_input_file : str
        Name of the find_orb input file.
    fo_output_folder : str
        Name of the find_orb output folder.
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
    """

    # Generate the find_orb commands
    fo_command = (
        f"cd {FO_DIR}; ./fo {fo_input_file} "
        f"-O {fo_output_folder}; cp -r {fo_output_folder} "
        f"{RUN_DIR}/{RESULT_DIR}/; cd {RUN_DIR}"
    )
    print(f"Find Orb command: {fo_command}")

    # Ensure the output directory exists and copy the input file
    os.makedirs(f"{FO_DIR}/{fo_output_folder}", exist_ok=True)
    shutil.copyfile(
        f"{RESULT_DIR}/{fo_input_file}",
        f"{FO_DIR}/{fo_input_file}",
    )

    # Run find_orb and check for output
    subprocess.run(fo_command, shell=True)
    if not os.path.exists(f"{FO_DIR}/{fo_output_folder}/covar.json"):
        print("No find_orb output for: ", fo_output_folder)
        return None
    else:
        # Convert to ADAM Orbit objects
        orbit = fo_to_adam_orbit_cov(f"{RESULT_DIR}/{fo_output_folder}")

        return orbit
