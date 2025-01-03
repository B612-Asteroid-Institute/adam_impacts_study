import logging
import os
import shutil
from typing import Iterator, Optional, Type

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
from adam_impact_study.types import ImpactStudyResults
from adam_impact_study.utils import get_study_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def run_impact_study_all(
    impactor_orbits: Orbits,
    population_config_file: str,
    pointing_file: str,
    base_dir: str,
    run_name: str,
    max_processes: Optional[int] = 1,
    overwrite: bool = True,
) -> Optional[ImpactStudyResults]:
    """
    Run an impact study for all impactors in the input file.

    Parameters
    ----------
    impactor_orbits : Orbits
        Orbits of the impactors to study
    population_config_file : str
        Path to the population config file
    pointing_file : str
        Path to the file containing pointing data for Sorcha.
    base_dir : str
        Base directory for all results
    run_name : str
        Name of the run.
    max_processes : int, optional
        Maximum number of processes to use for impact calculation (default: 1)
    overwrite : bool, optional
        Whether to overwrite existing run directory (default: True)

    Returns
    -------
    impact_results : ImpactStudyResults
        Table containing the results of the impact study with columns 'object_id',
        'day', and 'impact_probability'. If no impacts were found, returns None.

    """
    class ImpactASSISTPropagator(ASSISTPropagator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_dt = 1e-6
            self.min_dt = 1e-9
            self.adaptive_mode = 1
            self.epsilon = 1e-6

    # If the run directory already exists, throw an exception
    # unless the user has specified the overwrite flag
    if os.path.exists(f"{base_dir}/{run_name}"):
        if not overwrite:
            raise ValueError(f"Run directory {base_dir}/{run_name} already exists. Set overwrite=True to overwrite.")
        logger.warning(f"Overwriting run directory {base_dir}/{run_name}")
        shutil.rmtree(f"{base_dir}/{run_name}")

    os.makedirs(f"{base_dir}/{run_name}", exist_ok=True)

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
                ImpactASSISTPropagator,
                population_config_file,
                pointing_file,
                base_dir,
                run_name,
                max_processes=max_processes,
            )
            impact_results = qv.concatenate([impact_results, impact_result])
        else:
            futures.append(
                run_impact_study_fo_remote.remote(
                    impactor_orbit,
                    ImpactASSISTPropagator,
                    population_config_file,
                    pointing_file,
                    base_dir,
                    run_name,
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


def get_observation_windows(
    observations: Observations, impact_time: Timestamp, chunk_size: int
) -> Iterator[Observations]:

    min_mjd = pc.min(observations.coordinates.time.mjd())
    mask = pc.equal(observations.coordinates.time.mjd(), min_mjd)
    first_obs = observations.apply_mask(mask).coordinates.time

    # Initialize time to first observation
    day_count = first_obs
    while day_count.mjd()[0].as_py() < impact_time.mjd()[0].as_py():
        day_count = day_count.add_days(chunk_size)
        day = day_count.mjd()[0].as_py()
        logger.debug("Day: ", day)
        filtered_obs = observations.apply_mask(
            pc.less_equal(observations.coordinates.time.days.to_numpy(), day)
        )
        yield filtered_obs


def run_impact_study_fo(
    impactor_orbit: Orbits,
    propagator_class: Type[ASSISTPropagator],
    population_config_file: str,
    pointing_file: str,
    base_dir: str,
    run_name: str,
    max_processes: int = 1,
) -> ImpactStudyResults:
    """Run impact study for a single object"""
    assert (
        len(impactor_orbit.object_id) == 1
    ), "Impactor orbit must contain exactly one object"
    obj_id = impactor_orbit.object_id[0].as_py()
    logger.info(f"Processing object: {obj_id}")

    # Get paths for Sorcha
    paths = get_study_paths(base_dir, run_name, obj_id)

    # Run Sorcha to generate observations
    observations = run_sorcha(
        impactor_orbit,
        pointing_file,
        population_config_file,
        paths["sorcha_inputs"],
        paths["sorcha_outputs"],
        f"{run_name}_{obj_id}",
    )

    if len(observations) == 0:
        return ImpactStudyResults.empty()

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

    # Process each time window
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
                propagator_class,
                base_dir,
                run_name,
            )
            # Log if any error is present
            if pc.any(pc.invert(pc.is_null(result.error))).as_py():
                logger.warning(f"Error: {result.error}")
            results = qv.concatenate([results, result])
            if results.fragmented():
                results = qv.defragment(results)

        else:
            futures.append(
                calculate_impact_probability_remote.remote(
                    observations_window,
                    impactor_orbit,
                    propagator_class,
                    base_dir,
                    run_name,
                )
            )

            if len(futures) > max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                if pc.any(pc.invert(pc.is_null(result.error))).as_py():
                    logger.warning(f"Error: {result.error}")
                results = qv.concatenate([results, result])

    # Get remaining results
    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        if pc.any(pc.invert(pc.is_null(result.error))).as_py():
            logger.warning(f"Error: {result.error}")
        results = qv.concatenate([results, result])

    results.to_parquet(f"{paths['object_base_dir']}/impact_results_{obj_id}.parquet")

    return results


run_impact_study_fo_remote = ray.remote(run_impact_study_fo)


def calculate_impact_probability(
    observations: Observations,
    impactor_orbit: Orbits,
    propagator_class: Type[ASSISTPropagator],
    base_dir: str,
    run_name: str,
    max_processes: int = 1,
) -> ImpactStudyResults:
    """Calculate impact probability for a s of observations.

    Parameters
    ----------
    observations : Observations
        Observations to calculate an orbit from and determine impact probability.
    impactor_orbit : Orbits
        Original impactor orbit
    propagator_class : Type[ASSISTPropagator]
        Propagator class
    base_dir : str
        Base directory for all results
    run_name : str
        Name of the study run
    max_processes : int
        Maximum number of processes to use for impact calculation

    Returns
    -------
    ImpactStudyResults
        Impact probability results for this day if successful
    """
    obj_id = impactor_orbit.object_id[0].as_py()
    start_date = observations.coordinates.time.min()
    end_date = observations.coordinates.time.max()
    window_name = f"{start_date.mjd()[0]}_{end_date.mjd()[0]}"
    paths = get_study_paths(base_dir, run_name, obj_id, window_name)

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

    rejected_observations = ADESObservations.empty()

    try:
        orbit, rejected_observations, error = run_fo_od(
            observations,
            paths,
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
        propagator = propagator_class()
        propagated_30_days_before_impact = propagator.propagate_orbits(
            orbit,
            thirty_days_before_impact,
            covariance=True,
            covariance_method="monte-carlo",
            num_samples=1000,
        )
        propagated_30_days_before_impact.to_parquet(
            f"{paths['propagated']}/orbits.parquet"
        )
    except Exception as e:
        logger.error(f"Error propagating orbits: {e}")
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
        # Note: do we want to save the original variants here?
        propagator = propagator_class()
        final_orbit_states, impacts = calculate_impacts(
            propagated_30_days_before_impact,
            60,
            propagator,
            num_samples=10000,
            processes=max_processes,
        )
        final_orbit_states.to_parquet(
            f"{paths['propagated']}/monte_carlo_variant_states.parquet"
        )
        impacts.to_parquet(f"{paths['propagated']}/monte_carlo_impacts.parquet")

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
