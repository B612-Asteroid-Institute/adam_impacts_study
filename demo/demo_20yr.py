import os
import pandas as pd
import pyarrow.compute as pc
import numpy as np
import quivr as qv

from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.propagator.adam_assist import ASSISTPropagator

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.sorcha_utils import run_sorcha, sorcha_output_to_od_observations, write_config_file_timeframe
from adam_impact_study.conversions import impactor_file_to_adam_orbit, od_observations_to_ades_file
from adam_impact_study.impacts_study import ImpactStudyResults

# Define the run name and directories
RUN_NAME = "20yr_100"
RESULT_DIR = "100_test_results"
RUN_DIR = os.getcwd()
FO_DIR = "../find_orb/find_orb"

# Define the input files
impactors_file = "data/100_impactors.csv"
pointing_file = "data/twenty_nr_cycles_8_v3.3_20yrs.db"

physical_params_string = "15.88 1.72 0.48 -0.11 -0.12 -0.12 0.15"

# Run the impact study
propagator = ASSISTPropagator()

os.makedirs(f"{RESULT_DIR}", exist_ok=True)

# Read impactor data and convert to ADAM orbit objects
adam_orbit_objects = impactor_file_to_adam_orbit(impactors_file)

# Prepare physical parameters DataFrame
physical_params_list = [
    float(param) for param in physical_params_string.split()
]

print("Beginning impact study")

# Run Sorcha to generate observational data
for obj_id in adam_orbit_objects.object_id:

    # Additional file names generated from the run name
    sorcha_config_file_name = f"data/sorcha_config_{RUN_NAME}_{obj_id}.ini"
    sorcha_orbits_file = f"data/sorcha_input_{RUN_NAME}_{obj_id}.csv"
    sorcha_physical_params_file = f"data/sorcha_params_{RUN_NAME}_{obj_id}.csv"
    sorcha_output_name = f"sorcha_output_{RUN_NAME}_{obj_id}"
    sorcha_output_file = f"{sorcha_output_name}.csv"
    fo_input_file_base = f"fo_input_{RUN_NAME}_{obj_id}"
    fo_output_file_base = f"fo_output_{RUN_NAME}_{obj_id}"

    print("##############################")
    print("Starting object: ", obj_id)
    ip_results = None
    data = []
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
    orbit = adam_orbit_objects.apply_mask(pc.equal(adam_orbit_objects.object_id, obj_id))
    impact_date = orbit.coordinates.time.add_days(30)
    sorcha_config_file = write_config_file_timeframe(impact_date.mjd()[0], sorcha_config_file_name)

    print("Impact date: ", impact_date.mjd())
    print("Sorcha output file: ", sorcha_output_file)

    if not os.path.exists(f"{RESULT_DIR}/{sorcha_output_name}/{sorcha_output_file}"):

        try:
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
        except Exception as e:
            print(f"Error running Sorcha for {obj_id}: {e}")
            continue
    else:
        print(f"Skipping Sorcha run for {obj_id}")

    od_observations = sorcha_output_to_od_observations(f"{RESULT_DIR}/{sorcha_output_name}/{sorcha_output_file}")
    if od_observations is None:
        print(f"No observations for {obj_id}")
        continue
    print("OD Observations: ", od_observations)
    print(od_observations.object_id)

    print("Object ID: ", obj_id)
    od_obs = od_observations.apply_mask(pc.equal(od_observations.object_id, obj_id))
    print("Observations: ", od_obs)

    print(od_obs.coordinates.time.mjd())
    min_mjd = pc.min(od_obs.coordinates.time.mjd())
    mask = pc.equal(od_obs.coordinates.time.mjd(), min_mjd)
    start_date = od_obs.apply_mask(mask).coordinates.time

    day = start_date

    print("Start date: ", start_date.mjd())
    print("Impact date: ", impact_date.mjd())

    while day.mjd()[0].as_py() < impact_date.mjd()[0].as_py(): 
        print ("Time: ", day.mjd())
        day = day.add_days(365)
        print("Time: ", day.mjd())
        filtered_obs = od_obs.apply_mask(
            pc.less_equal(od_obs.coordinates.time.days.to_numpy(), day.mjd()[0].as_py())
        )
        print("Filtered Observations: ", filtered_obs)

        fo_file_name = f"{fo_input_file_base}_{obj_id}.csv"
        fo_output_folder = f"{fo_output_file_base}_{obj_id}"
        od_observations_to_ades_file(filtered_obs, f"{RESULT_DIR}/{fo_file_name}")

        try:
            fo_orbit = run_fo_od(
                fo_file_name,
                fo_output_folder,
                FO_DIR,
                RUN_DIR,
                RESULT_DIR,
            )
        except Exception as e:
            print(f"Error running find_orb output for {obj_id}: {e}")
            continue
        print(f"Fo orbit: {fo_orbit}")
        if fo_orbit is not None:
            print(f"Fo orbit elements: {fo_orbit.coordinates.values}")

        if fo_orbit is not None and len(fo_orbit) > 0:
            time = adam_orbit_objects.select("object_id", obj_id).coordinates.time[0]
            orbit = fo_orbit
            result = propagator.propagate_orbits(
                orbit, time, covariance=True, num_samples=1000
            )
            result.to_parquet(f"{RESULT_DIR}/testdatafortests_{obj_id}.parquet")
            print(f"Propagated orbit: {result}")
            print(f"Propagated orbit elements: {result.coordinates.values}")
            results, impacts = calculate_impacts(
                result, 60, propagator, num_samples=10000, processes=1
            )
            print(f"Impacts: {impacts}")
            print(f"Results: {results}")
            ip = calculate_impact_probabilities(results, impacts)
            print(f"IP: {ip.cumulative_probability[0].as_py()}")
            if ip.cumulative_probability[0].as_py() is not None:
                print(f"IP: {ip.cumulative_probability[0].as_py()}")

            if ip_results is None:
                ip_results = ImpactStudyResults.from_kwargs(
                    object_id=[obj_id],
                    day=[day.mjd()[0].as_py()],
                    impact_probability=[ip.cumulative_probability[0].as_py()],
                )
            else:
                ip_results = qv.concatenate([ip_results, ImpactStudyResults.from_kwargs(
                    object_id=[obj_id],
                    day=[day.mjd()[0].as_py()],
                    impact_probability=[ip.cumulative_probability[0].as_py()],
                )])
            print(ip_results)
            print(ip_results.day)
    
    plot_ip_over_time(ip_results)