import logging
import os
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_assist import ASSISTPropagator
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_core.observers.utils import calculate_observing_night
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                run_config_file,
                pointing_file,
                RUN_NAME,
                FO_DIR,
                RUN_DIR,
                RESULT_DIR,
                max_processes,
            )
            impact_results = qv.concatenate([impact_results, impact_result])
        else:
            futures.append(
                run_impact_study_fo_remote.remote(
                    impactor_orbit,
                    propagator,
                    run_config_file,
                    pointing_file,
                    RUN_NAME,
                    FO_DIR,
                    RUN_DIR,
                    RESULT_DIR,
                    max_processes,
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
    run_config_file: str,
    pointing_file: str,
    RUN_NAME: str,
    FO_DIR: str,
    RUN_DIR: str,
    RESULT_DIR: str,
    max_processes: int = 1,
) -> ImpactStudyResults:
    """Run impact study with optional parallel processing"""
    obj_id = impactor_orbit.object_id[0]
    logger.info(f"Object ID: {obj_id}")

    # Create object-specific result directory
    obj_result_dir = os.path.join(RESULT_DIR, f"{RUN_NAME}_{obj_id}")
    os.makedirs(obj_result_dir, exist_ok=True)

    # Create Sorcha object output directory
    sorcha_output_dir = os.path.join(RESULT_DIR, f"sorcha_output_{RUN_NAME}_{obj_id}")
    os.makedirs(sorcha_output_dir, exist_ok=True)

    # Define file paths relative to result directory
    sorcha_config_file_name = os.path.join(
        obj_result_dir, f"sorcha_config_{RUN_NAME}_{obj_id}.ini"
    )
    sorcha_orbits_file = os.path.join(
        obj_result_dir, f"sorcha_input_{RUN_NAME}_{obj_id}.csv"
    )
    sorcha_physical_params_file = os.path.join(
        obj_result_dir, f"sorcha_params_{RUN_NAME}_{obj_id}.csv"
    )
    sorcha_output_stem = f"{RUN_NAME}_{obj_id}"
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
    observations = run_sorcha(
        impactor_orbit,
        sorcha_config_file,
        sorcha_orbits_file,
        sorcha_physical_params_file,
        pointing_file,
        sorcha_output_dir,
        sorcha_output_stem,
    )
    if len(observations) == 0:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=Timestamp.from_mjd([0], scale="utc"),
            observation_end=Timestamp.from_mjd([0], scale="utc"),
            observation_count=[0],
            observation_nights=[0],
            error=["No observations recovered."],
        )

    # Sort the observations by time and origin code
    observations = observations.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos", "coordinates.origin.code"]
    )

    # Select the unique nights of observations and
    nights = calculate_observing_night(
        observations.coordinates.origin.code, observations.coordinates.time
    )
    unique_nights = pc.unique(nights).sort()

    if len(unique_nights) < 3:
        # TODO: We might consider returning something else here.
        return ImpactStudyResults.empty()

    # We iterate through unique nights and filter observations based on
    # to everything below or equal to the current night number
    # We start with a minimum of three unique nights
    futures = []
    results = ImpactStudyResults.empty()
    for night in unique_nights[2:]:
        mask = pc.less_equal(nights, night)
        observations_window = observations.apply_mask(mask)

        if max_processes == 1:
            result = calculate_impact_probability(
                observations_window,
                impactor_orbit,
                propagator,
                fo_input_file_base,
                FO_DIR,
                RUN_DIR,
                RESULT_DIR,
            )
            if result.error is not None:
                logger.info(f"Error: {result.error}")
            results = qv.concatenate([results, result])
            if results.fragmented():
                results = qv.defragment(results)

        else:
            futures.append(
                calculate_impact_probability_remote.remote(
                    observations_window,
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
                if result.error is not None:
                    logger.info(f"Error: {result.error}")
                results = qv.concatenate([results, result])

    # Get remaining results
    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        if result.error is not None:
            logger.info(f"Error: {result.error}")
        results = qv.concatenate([results, result])

    return results


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
    od_observations_to_ades_file(observations, f"{RESULT_DIR}/{fo_file_name}")

    try:
        orbit, error = run_fo_od(
            fo_file_name,
            obj_id,
            FO_DIR,
            RESULT_DIR,
        )
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            error=[str(e)],
        )

    if error is not None:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            error=[error],
        )

    try:
        propagated_30_days_before_impact = propagator.propagate_orbits(
            orbit,
            thirty_days_before_impact,
            covariance=True,
            covariance_method="monte-carlo",
            num_samples=1000,
        )
        propagated_30_days_before_impact.to_parquet(
            f"{RESULT_DIR}/propagated_orbit_{obj_id}_{start_date}_{end_date}.parquet"
        )
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            error=[str(e)],
        )

    try:
        final_orbit_states, impacts = calculate_impacts(
            propagated_30_days_before_impact, 60, propagator, num_samples=10000
        )

        ip = calculate_impact_probabilities(final_orbit_states, impacts)
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            object_id=[obj_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            error=[str(e)],
        )

    return ImpactStudyResults.from_kwargs(
        object_id=[obj_id],
        observation_start=start_date,
        observation_end=end_date,
        observation_count=[observations_count],
        observation_nights=[observation_nights],
        impact_probability=[ip.cumulative_probability[0].as_py()],
    )


# Create remote version
calculate_impact_probability_remote = ray.remote(calculate_impact_probability)
