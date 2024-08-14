import json
import os
import shutil
import subprocess

import quivr as qv
from conversions import fo_to_adam_orbit_cov, sorcha_df_to_fo_input


# Read the find_orb output files into a dictionary of orbits and covariances
def read_fo_output(fo_output_dir):
    covar_dict = read_fo_covariance(f"{fo_output_dir}/covar.json")
    elements_dict = read_fo_orbits(f"{fo_output_dir}/total.json")
    return elements_dict, covar_dict


# Read the find_orb covariance file into a dictionary
def read_fo_covariance(covar_file):
    with open(covar_file, "r") as f:
        covar_json = json.load(f)
    return covar_json


# Read the find_orb total.json file into a dictionary of orbital elements
def read_fo_orbits(input_file):
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
    Run the find_orb orbit determination.
    """
    # Read the find_orb output files into a dictionary of orbits and
    # covariances
    orbits = None
    for obj_id in sorcha_observations_df["ObjID"].unique():
        df = sorcha_observations_df[sorcha_observations_df["ObjID"] == obj_id]
        sorcha_df_to_fo_input(df, f"{RESULT_DIR}/{fo_input_file_base}_{obj_id}.csv")
        fo_output_folder = f"{fo_output_file_base}_{obj_id}"
        fo_command = (
            f"cd {FO_DIR}; ./fo {fo_input_file_base}_{obj_id}.csv "
            f"-O {fo_output_folder}; cp -r {fo_output_folder} "
            f"{RUN_DIR}/{RESULT_DIR}/; cd {RUN_DIR}"
        )
        os.makedirs(f"{FO_DIR}/{fo_output_folder}", exist_ok=True)
        shutil.copyfile(
            f"{RESULT_DIR}/{fo_input_file_base}_{obj_id}.csv",
            f"{FO_DIR}/{fo_input_file_base}_{obj_id}.csv",
        )
        subprocess.run(fo_command, shell=True)
        if not os.path.exists(f"{fo_output_file_base}_{obj_id}/covar.json"):
            print("No find_orb output for object: ", obj_id)
            continue
        elements_dict, covar_dict = read_fo_output(
            f"{RESULT_DIR}/{fo_output_file_base}_{obj_id}"
        )
        orbit = fo_to_adam_orbit_cov(elements_dict, covar_dict)[obj_id]
        if orbits is None:
            orbits = orbit
        else:
            orbits = qv.concatenate([orbits, orbit])
    return orbits
