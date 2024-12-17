import os
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_assist import ASSISTPropagator
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.orbit_determination import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from adam_impact_study.conversions import Observations, od_observations_to_ades_file
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.physical_params import (
    create_physical_params_single,
    photometric_properties_to_sorcha_table,
    write_phys_params_file,
)
from adam_impact_study.sorcha_utils import run_sorcha, write_config_file_timeframe


class ImpactStudyResults(qv.Table):
    object_id = qv.LargeStringColumn()
    observation_start = Timestamp.as_column()
    observation_end = Timestamp.as_column()
    observation_count = qv.UInt64Column()
    observation_nights = qv.UInt64Column()
    impact_probability = qv.Float64Column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)


def run_impact_study_all(
    impactor_orbits: Orbits,
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
    propagator = ASSISTPropagator(initial_dt=0.001, min_dt=1e-5, adaptive_mode=1, epsilon=1e-6)
    os.makedirs(f"{RESULT_DIR}", exist_ok=True)

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
    impactor_orbit: Orbits,
    propagator: ASSISTPropagator,
    run_config_file: str,
    pointing_file: str,
    RUN_NAME: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
    chunk_size: Optional[int] = 1,
    max_processes: int = 1,
) -> ImpactStudyResults:
    """Run impact study with optional parallel processing"""
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

    od_observations = od_observations.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

    # Propagate the orbit and calculate the impact probability over time
    impact_results = None
    futures = []
    
    min_mjd = pc.min(od_observations.coordinates.time.mjd())
    mask = pc.equal(od_observations.coordinates.time.mjd(), min_mjd)
    first_obs = od_observations.apply_mask(mask).coordinates.time

    # Initialize time to first observation
    day_count = first_obs

    print("Impact Date: ", impact_date)


    # Select the unique nights of od_observations and
    
    from adam_core.observers.utils import calculate_observing_night
    nights = calculate_observing_night(od_observations.coordinates.origin.code, od_observations.coordinates.time)
    unique_nights = set(nights)
    unique_nights = sorted(unique_nights)
    print("Unique Nights: ", unique_nights)

    # We iterate through unique nights and filter observations based on 
    # to everything below or equal to the current night number
    # We start with a minimum of three unique nights
    for night in unique_nights[2:]:
        mask = pc.less_equal(nights, night)
        od_observations_window = od_observations.apply_mask(mask)
        
        if max_processes == 1:
            result = calculate_impact_probability_for_day(
                od_observations_window,
                impactor_orbit,
                propagator,
                fo_input_file_base,
                fo_output_file_base,
                FO_DIR,
                RUN_DIR,
                RESULT_DIR,
            )
            if result is not None:
                impact_results = result if impact_results is None else qv.concatenate([impact_results, result])
        else:
            futures.append(
                calculate_impact_probability_for_day_remote.remote(
                    od_observations_window,
                    impactor_orbit,
                    propagator,
                    fo_input_file_base,
                    fo_output_file_base,
                    FO_DIR,
                    RUN_DIR,
                    RESULT_DIR,
                )
            )
            
            if len(futures) > max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                if result is not None:
                    impact_results = result if impact_results is None else qv.concatenate([impact_results, result])
                    
    # Get remaining results
    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        if result is not None:
            impact_results = result if impact_results is None else qv.concatenate([impact_results, result])
            
    return impact_results


def calculate_impact_probability_for_day(
    od_observations: Observations,
    impactor_orbit: Orbits,
    propagator: ASSISTPropagator,
    fo_input_file_base: str,
    fo_output_file_base: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
) -> Optional[ImpactStudyResults]:
    """Calculate impact probability for a specific day.
    
    Parameters
    ----------
    od_observations : Observations
        Filtered observations up to current day
    impactor_orbit : Orbits
        Original impactor orbit
    propagator : ASSISTPropagator
        Propagator instance
    fo_input_file_base : str
        Base filename for find_orb input
    fo_output_file_base : str
        Base filename for find_orb output
    FO_DIR : str
        Directory containing find_orb executable
    RUN_DIR : str
        Working directory
    RESULT_DIR : str
        Results output directory
        
    Returns
    -------
    ImpactStudyResults
        Impact probability results for this day if successful
    """
    obj_id = impactor_orbit.object_id[0].as_py()
    thirty_days_before_impact = impactor_orbit.coordinates.time

    start_date = od_observations.coordinates.time.min().mjd()[0].as_py()
    end_date = od_observations.coordinates.time.max().mjd()[0].as_py()

    fo_file_name = f"{fo_input_file_base}_{start_date}_{end_date}.csv"
    fo_output_folder = f"{fo_output_file_base}_{obj_id}_{start_date}_{end_date}"
    od_observations_to_ades_file(od_observations, f"{RESULT_DIR}/{fo_file_name}")

    try:
        orbit, error = run_fo_od(
            fo_file_name,
            fo_output_folder, 
            FO_DIR,
            RUN_DIR,
            RESULT_DIR,
        )
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=[od_observations.coordinates.time.min()],
            observation_end=[od_observations.coordinates.time.max()],
            error=[str(e)],
        )

    if error is not None:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=[od_observations.coordinates.time.min()],
            observation_end=[od_observations.coordinates.time.max()],
            error=[error],
        )

    # At this point we can guarante we at least have an orbit
    time = impactor_orbit.coordinates.time[0]
    try:
        propagated_30_days_from_impact = propagator.propagate_orbits(
            orbit, time, covariance=True, covariance_method="monte_carlo", num_samples=1000
        )
        propagated_30_days_from_impact.to_parquet(
            f"{RESULT_DIR}/propagated_orbit_{obj_id}_{start_date}_{end_date}.parquet"
        )
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=[od_observations.coordinates.time.min()],
            observation_end=[od_observations.coordinates.time.max()],
            error=[str(e)],
        )
        
    try:
        final_orbit_states, impacts = calculate_impacts(
            propagated_30_days_from_impact, 60, propagator, num_samples=10000
        )

        ip = calculate_impact_probabilities(final_orbit_states, impacts)
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=[od_observations.coordinates.time.min()],
            observation_end=[od_observations.coordinates.time.max()],
            error=[str(e)],
        )
    
    return ImpactStudyResults.from_kwargs(
        object_id=[obj_id],
        observation_start=[od_observations.coordinates.time.min()],
        observation_end=[od_observations.coordinates.time.max()],
        impact_probability=[ip.cumulative_probability[0].as_py()],
        observation_count=[len(od_observations)],
    )



# Create remote version
calculate_impact_probability_for_day_remote = ray.remote(calculate_impact_probability_for_day)
