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
    Run an impact study with the given impactors and confirguartion files.
    """

    propagator = ASSISTPropagator()

    impactor_df = pd.read_csv(impactors_file, float_precision="round_trip")
    initial_orbit_objects = impactor_to_adam_orbit(impactor_df)

    physical_params_list = [
        float(param) for param in sorcha_physical_params_string.split()
    ]

    data = []
    for obj_id in initial_orbit_objects.object_id:
        data.append(
            {
                # "ObjID": obj_id,
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
        print(ip_dict)
        ip_dict_obj_fo[obj] = ip_dict

    print(ip_dict_obj_fo)
