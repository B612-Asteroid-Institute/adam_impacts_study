import logging
import os
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_assist import ASSISTPropagator
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.observations.ades import ADESObservations
from adam_core.observers.utils import calculate_observing_night
from adam_core.orbit_determination import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.conversions import Observations, od_observations_to_ades_file
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.physical_params import (
    create_physical_params_single,
    photometric_properties_to_sorcha_table,
    write_phys_params_file,
)
from adam_impact_study.sorcha_utils import run_sorcha, write_config_file_timeframe
from adam_impact_study.utils import get_study_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImpactStudyResults(qv.Table):
    object_id = qv.LargeStringColumn()
    observation_start = Timestamp.as_column()
    observation_end = Timestamp.as_column()
    observation_count = qv.UInt64Column()
    observations_rejected = qv.UInt64Column()
    observation_nights = qv.UInt64Column()
    impact_probability = qv.Float64Column(nullable=True)

    error = qv.LargeStringColumn(nullable=True)


def run_impact_study_all(
    impactor_orbits: Orbits,
    population_config_file: str,
    pointing_file: str,
    RUN_NAME: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
    max_processes: Optional[int] = 1,
) -> Optional[ImpactStudyResults]:
    """
    Run an impact study for all impactors in the input file.

    Parameters
    ----------
    impactors_file : str
        Path to the CSV file containing impactor data.
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

    Returns
    -------
    impact_results : ImpactStudyResults
        Table containing the results of the impact study with columns 'object_id',
        'day', and 'impact_probability'. If no impacts were found, returns None.

    """
    propagator = ASSISTPropagator(
        initial_dt=0.001, min_dt=1e-12, adaptive_mode=1, epsilon=1e-9
    )
    os.makedirs(f"{RESULT_DIR}", exist_ok=True)

    logger.info(f"Impactor Orbits: {impactor_orbits}")
    object_ids = impactor_orbits.object_id.unique()
    logger.info(f"Object IDs: {object_ids.to_pylist()}")

    impact_results = ImpactStudyResults.empty()

    futures = []
    for obj_id in object_ids:
        impactor_orbit = impactor_orbits.select("object_id", obj_id)

        if max_processes == 1:
            impact_result = run_impact_study_fo(
                impactor_orbit,
                propagator,
                population_config_file,
                pointing_file,
                RUN_DIR,
                RUN_NAME,
                FO_DIR,
                max_processes=max_processes,
            )
            impact_results = qv.concatenate([impact_results, impact_result])
        else:
            futures.append(
                run_impact_study_fo_remote.remote(
                    impactor_orbit,
                    propagator,
                    population_config_file,
                    pointing_file,
                    RUN_DIR,
                    RUN_NAME,
                    FO_DIR,
                    max_processes=max_processes,
                )
            )

            if len(futures) > max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                impact_results = qv.concatenate([impact_results, result])

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        impact_results = qv.concatenate([impact_results, result])

    return impact_results


def run_impact_study_fo(
    impactor_orbit: Orbits,
    propagator: ASSISTPropagator,
    population_config_file: str,
    pointing_file: str,
    base_dir: str,
    run_name: str,
    fo_dir: str,
    max_processes: int = 1,
) -> ImpactStudyResults:
    """Run impact study for a single object"""
    assert len(impactor_orbit.object_id) == 1, "Impactor orbit must contain exactly one object"
    obj_id = impactor_orbit.object_id[0].as_py()
    logger.info(f"Processing object: {obj_id}")
    
    # Get paths for Sorcha
    paths = get_study_paths(base_dir, run_name, obj_id)
    
    # Run Sorcha to generate observations
    observations = run_sorcha(
        impactor_orbit,
        population_config_file,
        pointing_file,
        paths['sorcha_inputs'],
        paths['sorcha_outputs'],
        f"{run_name}_{obj_id}"
    )
    
    if len(observations) == 0:
        return ImpactStudyResults.empty()

    # Process each time window
    results = []
    for obs_window in get_observation_windows(observations):
        start_mjd = obs_window.coordinates.time.mjd()[0]
        end_mjd = obs_window.coordinates.time.mjd()[-1]
        time_range = f"{start_mjd}__{end_mjd}"
        
        # Get paths for this time window
        window_paths = get_study_paths(base_dir, run_name, obj_id, time_range)
        
        result = process_observation_window(
            obs_window,
            impactor_orbit,
            propagator,
            fo_dir,
            window_paths
        )
        results.append(result)
    
    return qv.concatenate(results) if results else ImpactStudyResults.empty()


run_impact_study_fo_remote = ray.remote(run_impact_study_fo)


def calculate_impact_probability(
    observations: Observations,
    impactor_orbit: Orbits,
    propagator: ASSISTPropagator,
    fo_input_file_base: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
) -> ImpactStudyResults:
    """Calculate impact probability for a s of observations.

    Parameters
    ----------
    observations : Observations
        Observations to calculate an orbit from and determine impact probability.
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

    # Get the start and end date of the observations, the number of
    # observations, and the number of unique nights
    start_date = observations.coordinates.time.min()
    end_date = observations.coordinates.time.max()
    observations_count = len(observations)
    nights = calculate_observing_night(
        observations.coordinates.origin.code, observations.coordinates.time
    )
    unique_nights = pc.unique(nights).sort()
    observation_nights = len(unique_nights)

    # Create the find_orb input and output files
    start_date_mjd = start_date.mjd()[0]
    end_date_mjd = end_date.mjd()[0]
    fo_file_name = f"{fo_input_file_base}_{start_date_mjd}_{end_date_mjd}.csv"
    fo_file_path = f"{RESULT_DIR}/{fo_file_name}"
    od_observations_to_ades_file(observations, f"{RESULT_DIR}/{fo_file_name}")

    # Create a unique run name based on the object ID and the start and end date
    run_name = f"{obj_id}_{start_date_mjd}_{end_date_mjd}"

    rejected_observations = ADESObservations.empty()

    try:
        orbit, rejected_observations, error = run_fo_od(
            FO_DIR,
            RESULT_DIR,
            fo_file_path,
            run_name,
        )
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[str(e)],
        )

    if error is not None:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[error],
        )

    try:
        propagated_30_days_before_impact = propagator.propagate_orbits(
            orbit,
            thirty_days_before_impact,
            covariance=True,
            covariance_method="monte-carlo",
            #covariance_representation="keplerian", Would sample elements from keplerian space
            num_samples=1000,
        )
        propagated_30_days_before_impact.to_parquet(
            f"{RESULT_DIR}/propagated_orbit_{obj_id}_{start_date.mjd()[0]}_{end_date.mjd()[0]}.parquet"
        )
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[str(e)],
        )

    try:
        #Note: do we want to save the original variants here?
        final_orbit_states, impacts = calculate_impacts(
            propagated_30_days_before_impact, 60, propagator, num_samples=10000, #covariance_representation="keplerian" ??
        )
        final_orbit_states.to_parquet( 
            f"{RESULT_DIR}/monte_carlo_variant_states_{obj_id}_{start_date.mjd()[0]}_{end_date.mjd()[0]}.parquet"
        )
        impacts.to_parquet(
            f"{RESULT_DIR}/monte_carlo_impacts_{obj_id}_{start_date.mjd()[0]}_{end_date.mjd()[0]}.parquet"
        )

        ip = calculate_impact_probabilities(final_orbit_states, impacts)
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[str(e)],
        )

    return ImpactStudyResults.from_kwargs(
        object_id=[obj_id],
        observation_start=start_date,
        observation_end=end_date,
        observation_count=[observations_count],
        observation_nights=[observation_nights],
        observations_rejected=[len(rejected_observations)],
        impact_probability=[ip.cumulative_probability[0].as_py()],
    )


# Create remote version
calculate_impact_probability_remote = ray.remote(calculate_impact_probability)
