import numpy as np
import pandas as pd
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.propagator.adam_assist import ASSISTPropagator
from conversions import (
    impactor_to_adam_orbit,
    sorcha_output_to_df,
    sorcha_output_to_od_observations,
)
from fo_od import run_fo_od, sorcha_df_to_fo_input
from sorcha_utils import run_sorcha


def run_impact_study_fo(
    impactors_file,
    sorcha_config_file,
    sorcha_orbits_file,
    sorcha_physical_params_file,
    sorcha_output_file,
    sorcha_physical_params_string,
    pointing_file,
    sorcha_output_name,
    fo_input_file_base,
    fo_output_file_base,
    FO_DIR,
    RUN_DIR,
    RESULT_DIR,
):
    """
    Run an impact study using the given impactors and configuration files.

    This function performs an impact probability analysis by first converting
    impactor data into ADAM orbit objects, then running Sorcha to generate
    observation data. The observation data is then processed using find_orb
    and propagated to compute impact probabilities.

    Parameters
    ----------
    impactors_file : str
        Path to the CSV file containing impactor data.
    sorcha_config_file : str
        Path to the Sorcha configuration file.
    sorcha_orbits_file : str
        Path to the file where Sorcha orbit data will be saved.
    sorcha_physical_params_file : str
        Path to the file where Sorcha physical parameters will be saved.
    sorcha_output_file : str
        Name of the Sorcha output file.
    sorcha_physical_params_string : str
        String of space-separated physical parameters to be used in Sorcha.
    pointing_file : str
        Path to the file containing pointing data for Sorcha.
    sorcha_output_name : str
        Name for the output directory where Sorcha results will be saved.
    fo_input_file_base : str
        Base name for the find_orb input files.
    fo_output_file_base : str
        Base name for the find_orb output directories.
    FO_DIR : str
        Directory path where the find_orb executable is located.
    RUN_DIR : str
        Directory path where the script is being run.
    RESULT_DIR : str
        Directory where the results will be stored.

    Returns
    -------
    None
        The function prints the impact probability results for each object.
    """

    propagator = ASSISTPropagator()

    # Read impactor data and convert to ADAM orbit objects
    impactor_df = pd.read_csv(impactors_file, float_precision="round_trip")
    initial_orbit_objects = impactor_to_adam_orbit(impactor_df)

    # Prepare physical parameters DataFrame
    physical_params_list = [
        float(param) for param in sorcha_physical_params_string.split()
    ]
    data = []
    for obj_id in initial_orbit_objects.object_id:
        data.append(
            {
                "ObjID": str(obj_id),
                "H_r": physical_params_list[0],
                "u-r": physical_params_list[1],
                "g-r": physical_params_list[2],
                "i-r": physical_params_list[3],
                "z-r": physical_params_list[4],
                "y-r": physical_params_list[5],
                "GS": physical_params_list[6],
            }
        )
    physical_params_df = pd.DataFrame(data)

    # Run Sorcha to generate observational data
    sorcha_observations_df = run_sorcha(
        impactor_df,
        sorcha_config_file,
        sorcha_orbits_file,
        sorcha_physical_params_file,
        sorcha_output_file,
        physical_params_df,
        pointing_file,
        sorcha_output_name,
        RESULT_DIR,
    )
    sorcha_observations_df = sorcha_output_to_df(
        f"{sorcha_output_name}/{sorcha_output_file}"
    )
    od_observations = sorcha_output_to_od_observations(sorcha_observations_df)

    # Iterate over each object and calculate impact probabilities
    object_ids = od_observations.keys()
    ip_dict_obj_fo = {}
    for obj in object_ids:
        ip_dict = {}
        print("Object ID: ", obj)
        df = sorcha_observations_df[sorcha_observations_df["ObjID"] == obj]
        unique_days = np.floor(df["fieldMJD_TAI"]).unique()
        
        for day in unique_days:
            day = int(day)
            print("Day: ", day)
            filtered_df = df[np.floor(df["fieldMJD_TAI"]) <= day]
            fo_file_name = f"{fo_input_file_base}_{day}"
            fo_output_folder = f"{fo_output_file_base}_{day}"
            sorcha_df_to_fo_input(filtered_df, f"{RESULT_DIR}/{fo_file_name}")
            
            try:
                # Run find_orb to compute orbits
                fo_orbit = run_fo_od(
                    filtered_df,
                    fo_file_name,
                    fo_output_folder,
                    FO_DIR,
                    RUN_DIR,
                    RESULT_DIR,
                )
            except Exception as e:
                print(f"Error running find_orb output for {obj}: {e}")
                continue
            
            if len(fo_orbit) > 0:
                time = initial_orbit_objects.select("object_id", obj).coordinates.time[
                    0
                ]
                orbit = fo_orbit
                try:
                    # Propagate orbits and calculate impact probabilities
                    result = propagator.propagate_orbits(
                        orbit, time, covariance=True, num_samples=1000
                    )
                    results, impacts = calculate_impacts(
                        result, 60, propagator, num_samples=10000
                    )
                    ip = calculate_impact_probabilities(results, impacts)
                    ip_dict[day] = ip.cumulative_probability[0].as_py()
                    print(f"Impact Probability: {ip.cumulative_probability[0].as_py()}")
                except Exception as e:
                    print(f"Error calculating impacts for {obj}: {e}")
                    continue
        
        ip_dict_obj_fo[obj] = ip_dict

    return ip_dict_obj_fo