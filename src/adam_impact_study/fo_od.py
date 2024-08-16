import json
import os
import shutil
import subprocess

import quivr as qv
from adam_impact_study.conversions import fo_to_adam_orbit_cov, sorcha_df_to_fo_input


def read_fo_output(fo_output_dir):
    """
    Read the find_orb output files from the specified directory into dictionaries
    containing orbital elements and covariances.

    Parameters
    ----------
    fo_output_dir : str
        Directory path where find_orb output files (e.g., total.json and covar.json) are located.

    Returns
    -------
    elements_dict : dict
        Dictionary containing orbital elements for each object.
    covar_dict : dict
        Dictionary containing covariance matrices for each object.
    """
    covar_dict = read_fo_covariance(f"{fo_output_dir}/covar.json")
    elements_dict = read_fo_orbits(f"{fo_output_dir}/total.json")
    return elements_dict, covar_dict


def read_fo_covariance(covar_file):
    """
    Read the find_orb covariance JSON file into a dictionary.

    Parameters
    ----------
    covar_file : str
        Path to the find_orb covariance JSON file (covar.json).

    Returns
    -------
    covar_json : dict
        Dictionary containing the covariance data from the JSON file.
    """
    with open(covar_file, "r") as f:
        covar_json = json.load(f)
    return covar_json


def read_fo_orbits(input_file):
    """
    Read the find_orb total.json file into a dictionary of orbital elements.

    Parameters
    ----------
    input_file : str
        Path to the find_orb total.json file.

    Returns
    -------
    elements_dict : dict
        Dictionary containing orbital elements for each object.
    """
    with open(input_file, "r") as f:
        total_json = json.load(f)
    objects = total_json.get("objects", {})
    elements_dict = {}
    for object_id, object_data in objects.items():
        elements = object_data.get("elements", {})
        elements_dict[object_id] = elements
    return elements_dict


def run_fo_od(
    sorcha_observations_df,
    fo_input_file_base,
    fo_output_file_base,
    FO_DIR,
    RUN_DIR,
    RESULT_DIR,
):
    """
    Run the find_orb orbit determination process for each object

    Parameters
    ----------
    sorcha_observations_df : pandas.DataFrame
        DataFrame containing Sorcha observations
    fo_input_file_base : str
        Base name for the find_orb input files to be created.
    fo_output_file_base : str
        Base name for the find_orb output folders.
    FO_DIR : str
        Directory path where the find_orb executable is located.
    RUN_DIR : str
        Directory path where the script is being run.
    RESULT_DIR : str
        Directory path where the results will be stored.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Concatenated ADAM Orbit object containing orbits for all objects.
    """
    orbits = None
    for obj_id in sorcha_observations_df["ObjID"].unique():

        # Filter the observations for the current object and generate inputs
        df = sorcha_observations_df[sorcha_observations_df["ObjID"] == obj_id]
        sorcha_df_to_fo_input(df, f"{RESULT_DIR}/{fo_input_file_base}_{obj_id}.csv")
        fo_output_folder = f"{fo_output_file_base}_{obj_id}"

        # Generate the find_orb commands
        fo_command = (
            f"cd {FO_DIR}; ./fo {fo_input_file_base}_{obj_id}.csv "
            f"-O {fo_output_folder}; cp -r {fo_output_folder} "
            f"{RUN_DIR}/{RESULT_DIR}/; cd {RUN_DIR}"
        )
        print(f"Find Orb command: {fo_command}")

        # Ensure the output directory exists and copy the input file
        os.makedirs(f"{FO_DIR}/{fo_output_folder}", exist_ok=True)
        shutil.copyfile(
            f"{RESULT_DIR}/{fo_input_file_base}_{obj_id}.csv",
            f"{FO_DIR}/{fo_input_file_base}_{obj_id}.csv",
        )

        # Run find_orb and check for output
        subprocess.run(fo_command, shell=True)
        if not os.path.exists(f"{FO_DIR}/{fo_output_file_base}_{obj_id}/covar.json"):
            print("No find_orb output for object: ", obj_id)
            continue
        elements_dict, covar_dict = read_fo_output(
            f"{RESULT_DIR}/{fo_output_file_base}_{obj_id}"
        )

        # Convert to ADAM Orbit objects
        orbit = fo_to_adam_orbit_cov(elements_dict, covar_dict)[obj_id]
        if orbits is None:
            orbits = orbit
        else:
            orbits = qv.concatenate([orbits, orbit])

    return orbits
