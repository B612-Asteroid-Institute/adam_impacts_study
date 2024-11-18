import os
from typing import Optional

import pyarrow.compute as pc
import quivr as qv
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.propagator.adam_assist import ASSISTPropagator

from adam_impact_study.conversions import (
    impactor_file_to_adam_orbit,
    od_observations_to_ades_file,
)
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.physical_params import (
    create_physical_params_single,
    photometric_properties_to_sorcha_table,
    write_phys_params_file,
)
from adam_impact_study.sorcha_utils import run_sorcha, write_config_file_timeframe


class ImpactStudyResults(qv.Table):
    object_id = qv.LargeStringColumn()
    day = qv.Float64Column()
    impact_probability = qv.Float64Column()


def run_impact_study_all(
    impactors_file: str,
    run_config_file: str,
    pointing_file: str,
    RUN_NAME: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
    chunk_size: Optional[int] = 1,
) -> Optional[ImpactStudyResults]:
    """
    Run an impact study for all impactors in the input file.

    Parameters
    ----------
    impactors_file : str
        Path to the CSV file containing impactor data.
    sorcha_physical_params_string : str
        String containing the physical parameters for the impactors.
    pointing_file : str
        Path to the file containing pointing data for Sorcha.
    RUN_NAME : str
        Name of the run.
    FO_DIR : str
        Directory path where the find_orb executable is located.
    RUN_DIR : str
        Directory path where the script is being run.
    RESULT_DIR : str
        Directory where the results will be stored.
    chunk_size : int, optional
        Number of days to propagate orbits at a time.

    Returns
    -------
    impact_results : ImpactStudyResults
        Table containing the results of the impact study with columns 'object_id',
        'day', and 'impact_probability'. If no impacts were found, returns None.
    """
    propagator = ASSISTPropagator()
    os.makedirs(f"{RESULT_DIR}", exist_ok=True)

    impactor_orbits = impactor_file_to_adam_orbit(impactors_file)
    print("Impactor Orbits: ", impactor_orbits)
    object_ids = impactor_orbits.object_id.unique()
    print("Object IDs: ", object_ids)

    impact_results = None

    for obj_id in object_ids:
        impactor_orbit = impactor_orbits.apply_mask(
            pc.equal(impactor_orbits.object_id, obj_id)
        )
        impact_result = run_impact_study_fo(
            impactor_orbit,
            propagator,
            run_config_file,
            pointing_file,
            RUN_NAME,
            FO_DIR,
            RUN_DIR,
            RESULT_DIR,
            chunk_size,
        )

        if impact_results is None:
            impact_results = impact_result
        else:
            impact_results = qv.concatenate([impact_results, impact_result])

    return impact_results


def run_impact_study_fo(
    impactor_orbit: qv.Table,
    propagator: ASSISTPropagator,
    run_config_file: str,
    pointing_file: str,
    RUN_NAME: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
    chunk_size: Optional[int] = 1,
) -> Optional[ImpactStudyResults]:
    """
    Run a single impact study using find_orb to propagate orbits and calculate impact probabilities.


    Parameters
    ----------
    impactor_orbit : qv.Table
        Table containing the initial orbits of the impactors.
    propagator : ASSISTPropagator
        Propagator object to propagate orbits.
    sorcha_physical_params_string : str
        String containing the physical parameters for the impactors.
    pointing_file : str
        Path to the file containing pointing data for Sorcha.
    RUN_NAME : str
        Name of the run.
    FO_DIR : str
        Directory path where the find_orb executable is located.
    RUN_DIR : str
        Directory path where the script is being run.
    RESULT_DIR : str
        Directory where the results will be stored.
    chunk_size : int, optional
        Number of days to propagate orbits at a time, by default 1.

    Returns
    -------
    impact_results : ImpactStudyResults
        Table containing the results of the impact study with columns 'object_id',
        'day', and 'impact_probability'. If no impacts were found, returns None.
    """
    obj_id = impactor_orbit.object_id[0]
    print("Object ID: ", obj_id)

    sorcha_config_file_name = f"sorcha_config_{RUN_NAME}_{obj_id}.ini"
    sorcha_orbits_file = f"sorcha_input_{RUN_NAME}_{obj_id}.csv"
    sorcha_physical_params_file = f"sorcha_params_{RUN_NAME}_{obj_id}.csv"
    sorcha_output_name = f"sorcha_output_{RUN_NAME}_{obj_id}"
    sorcha_output_file = f"{sorcha_output_name}.csv"
    fo_input_file_base = f"fo_input_{RUN_NAME}_{obj_id}"
    fo_output_file_base = f"fo_output_{RUN_NAME}_{obj_id}"

    phys_params = create_physical_params_single(run_config_file, obj_id)
    phys_para_file_str = photometric_properties_to_sorcha_table(phys_params, "r")
    write_phys_params_file(phys_para_file_str, sorcha_physical_params_file)

    impact_date = impactor_orbit.coordinates.time.add_days(30)
    sorcha_config_file = write_config_file_timeframe(
        impact_date.mjd()[0], sorcha_config_file_name
    )

    # Run Sorcha to generate observational data
    od_observations = run_sorcha(
        impactor_orbit,
        sorcha_config_file,
        sorcha_orbits_file,
        sorcha_physical_params_file,
        sorcha_output_file,
        pointing_file,
        sorcha_output_name,
        RESULT_DIR,
    )
    if od_observations is None:
        return None

    # Propagate the orbit and calculate the impact probability over time
    impact_results = None

    min_mjd = pc.min(od_observations.coordinates.time.mjd())
    mask = pc.equal(od_observations.coordinates.time.mjd(), min_mjd)
    first_obs = od_observations.apply_mask(mask).coordinates.time

    # Initialize time to first observation
    day_count = first_obs

    print("Impact Date: ", impact_date)

    while day_count.mjd()[0].as_py() < impact_date.mjd()[0].as_py():
        day_count = day_count.add_days(chunk_size)
        day = day_count.mjd()[0].as_py()
        print("Day: ", day)
        filtered_obs = od_observations.apply_mask(
            pc.less_equal(od_observations.coordinates.time.days.to_numpy(), day)
        )
        print("Filtered Observations: ", filtered_obs)
        print("Filtered Days: ", filtered_obs.coordinates.time.days.to_numpy())

        fo_file_name = f"{fo_input_file_base}_{day}.csv"
        fo_output_folder = f"{fo_output_file_base}_{obj_id}_{day}"
        od_observations_to_ades_file(filtered_obs, f"{RESULT_DIR}/{fo_file_name}")

        # Run find_orb to compute orbits
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
            time = impactor_orbit.coordinates.time[0]
            print(f"Time: {time.mjd()}")
            orbit = fo_orbit
            try:
                # Propagate orbits and calculate impact probabilities
                result = propagator.propagate_orbits(
                    orbit, time, covariance=True, num_samples=1000
                )
                print(f"Propagated orbit: {result}")
                print(f"Propagated orbit elements: {result.coordinates.values}")
            except Exception as e:
                print(f"Error propagating orbits for {obj_id}: {e}")
                continue
            try:
                results, impacts = calculate_impacts(
                    result, 60, propagator, num_samples=10000
                )
                result.to_parquet(
                    f"{RESULT_DIR}/propagated_orbit_{obj_id}_{day}.parquet"
                )
                print(f"Impacts: {impacts}")
                ip = calculate_impact_probabilities(results, impacts)
                print(f"IP: {ip.cumulative_probability[0].as_py()}")
            except Exception as e:
                print(f"Error calculating impacts for {obj_id}: {e}")
                continue
            if ip.cumulative_probability[0].as_py() is not None:
                impact_result = ImpactStudyResults.from_kwargs(
                    object_id=[obj_id],
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
