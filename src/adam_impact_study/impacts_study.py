from typing import Optional

import os
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import quivr as qv
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.propagator.adam_assist import ASSISTPropagator

from adam_impact_study.conversions import (
    impactor_file_to_adam_orbit,
    od_observations_to_ades_file,
    od_observations_to_fo_input,
)
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.sorcha_utils import run_sorcha


class ImpactStudyResults(qv.Table):
    object_id = qv.LargeStringColumn()
    day = qv.Float64Column()
    impact_probability = qv.Float64Column()


def run_impact_study_fo(
    impactors_file: str,
    sorcha_config_file: str,
    sorcha_orbits_file: str,
    sorcha_physical_params_file: str,
    sorcha_output_file: str,
    sorcha_physical_params_string: str,
    pointing_file: str,
    sorcha_output_name: str,
    fo_input_file_base: str,
    fo_output_file_base: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
) -> Optional[ImpactStudyResults]:
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
    impact_results : ImpactStudyResults
        Table containing the results of the impact study with columns 'object_id',
        'day', and 'impact_probability'. If no impacts were found, returns None.
    """

    propagator = ASSISTPropagator()

    os.makedirs(f"{RESULT_DIR}", exist_ok=True)

    # Read impactor data and convert to ADAM orbit objects
    adam_orbit_objects = impactor_file_to_adam_orbit(impactors_file)

    # Prepare physical parameters DataFrame
    physical_params_list = [
        float(param) for param in sorcha_physical_params_string.split()
    ]
    data = []
    for obj_id in adam_orbit_objects.object_id:
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
    od_observations = run_sorcha(
        adam_orbit_objects,
        sorcha_config_file,
        sorcha_orbits_file,
        sorcha_physical_params_file,
        sorcha_output_file,
        physical_params_df,
        pointing_file,
        sorcha_output_name,
        RESULT_DIR,
    )
    if od_observations is None:
        return None

    # Iterate over each object and calculate impact probabilities
    object_ids = od_observations.object_id.unique()
    impact_results = None
    for obj in object_ids:
        print("Object ID: ", obj)
        od_obs = od_observations.apply_mask(pc.equal(od_observations.object_id, obj))
        days = od_obs.coordinates.time.days.to_numpy()
        unique_days = np.unique(days)
        for day in unique_days:
            print("Day: ", day)
            filtered_obs = od_obs.apply_mask(
                pc.less_equal(od_obs.coordinates.time.days.to_numpy(), day)
            )
            print("Filtered Observations: ", filtered_obs)
            print("Filtered Days: ", filtered_obs.coordinates.time.days.to_numpy())

            fo_file_name = f"{fo_input_file_base}_{obj}_{day}.csv"
            fo_output_folder = f"{fo_output_file_base}_{obj}_{day}"
            od_observations_to_ades_file(filtered_obs, f"{RESULT_DIR}/{fo_file_name}")

            try:
                # Run find_orb to compute orbits
                fo_orbit = run_fo_od(
                    fo_file_name,
                    fo_output_folder,
                    FO_DIR,
                    RUN_DIR,
                    RESULT_DIR,
                )
            except Exception as e:
                print(f"Error running find_orb output for {obj}: {e}")
                continue
            print(f"Fo orbit: {fo_orbit}")
            if fo_orbit is not None:
                print(f"Fo orbit elements: {fo_orbit.coordinates.values}")

            if fo_orbit is not None and len(fo_orbit) > 0:
                time = adam_orbit_objects.select("object_id", obj).coordinates.time[0]
                orbit = fo_orbit
                try:
                    # Propagate orbits and calculate impact probabilities
                    result = propagator.propagate_orbits(
                        orbit, time, covariance=True, num_samples=1000
                    )
                    print(f"Propagated orbit: {result}")
                    print(f"Propagated orbit elements: {result.coordinates.values}")
                    results, impacts = calculate_impacts(
                        result, 60, propagator, num_samples=10000
                    )
                    print(f"Impacts: {impacts}")
                    ip = calculate_impact_probabilities(results, impacts)
                    print(f"IP: {ip.cumulative_probability[0].as_py()}")
                except Exception as e:
                    print(f"Error calculating impacts for {obj}: {e}")
                    continue
                if ip.cumulative_probability[0].as_py() is not None:
                    impact_result = ImpactStudyResults.from_kwargs(
                        object_id=[obj],
                        day=[day],
                        impact_probability=[ip.cumulative_probability[0].as_py()],
                    )
                    print(f"Impact Result: {impact_result}")
                    if impact_results is None:
                        impact_results = impact_result
                    else:
                        impact_results = qv.concatenate([impact_results, impact_result])
                    print(f"Impact Results: {impact_results}")

    return impact_results
