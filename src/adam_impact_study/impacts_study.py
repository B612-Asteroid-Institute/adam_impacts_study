import logging
import os
import shutil
from typing import Iterator, Optional, Type

import numpy as np
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

from adam_impact_study.conversions import Observations
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.sorcha_utils import run_sorcha
from adam_impact_study.types import ImpactorOrbits, ImpactStudyResults
from adam_impact_study.utils import get_study_paths

from .utils import seed_from_string

logger = logging.getLogger(__name__)


def run_impact_study_all(
    impactor_orbits: ImpactorOrbits,
    pointing_file: str,
    run_dir: str,
    max_processes: Optional[int] = 1,
    overwrite: bool = True,
    seed: Optional[int] = 13612,
) -> Optional[ImpactStudyResults]:
    """
    Run an impact study for all impactors in the input file.

    Parameters
    ----------
    impactor_orbits : ImpactorOrbits
        Orbits of the impactors to study
    pointing_file : str
        Path to the file containing pointing data for Sorcha.
    run_dir : str
        Directory for this specific study run
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
    if os.path.exists(f"{run_dir}"):
        if not overwrite:
            raise ValueError(
                f"Run directory {run_dir} already exists. Set overwrite=True to overwrite."
            )
        logger.warning(f"Overwriting run directory {run_dir}")
        shutil.rmtree(f"{run_dir}")

    os.makedirs(f"{run_dir}", exist_ok=True)

    logger.info(f"Impactor Orbits: {impactor_orbits}")
    orbit_ids = impactor_orbits.orbit_id.unique()
    logger.info(f"Object IDs: {orbit_ids.to_pylist()}")

    impact_results = ImpactStudyResults.empty()

    futures = []
    for orbit_id in orbit_ids:
        impactor_orbit = impactor_orbits.select("orbit_id", orbit_id)

        orbit_seed = seed_from_string(orbit_id.as_py(), seed)

        if max_processes == 1:
            impact_result = run_impact_study_fo(
                impactor_orbit,
                ImpactASSISTPropagator,
                pointing_file,
                run_dir,
                max_processes=max_processes,
                seed=orbit_seed,
            )
            impact_results = qv.concatenate([impact_results, impact_result])
        else:
            futures.append(
                run_impact_study_fo_remote.remote(
                    impactor_orbit,
                    ImpactASSISTPropagator,
                    pointing_file,
                    run_dir,
                    max_processes=max_processes,
                    seed=orbit_seed,
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
    impactor_orbit: ImpactorOrbits,
    propagator_class: Type[ASSISTPropagator],
    pointing_file: str,
    run_dir: str,
    max_processes: Optional[int] = 1,
    seed: Optional[int] = None,
) -> ImpactStudyResults:
    """Run an impact study for a single impactor.

    Parameters
    ----------
    impactor_orbit : ImpactorOrbits
        Orbit of the impactor to study
    propagator_class : Type[ASSISTPropagator]
        Class to use for propagation
    pointing_file : str
        Path to the file containing pointing data for Sorcha
    run_dir : str
        Directory for this study run
    max_processes : Optional[int]
        Maximum number of processes to use for impact calculation

    Returns
    -------
    ImpactStudyResults
        Table containing the results of the impact study
    """
    assert len(impactor_orbit) == 1, "Only one object supported at a time"
    orbit_id = impactor_orbit.orbit_id[0].as_py()

    paths = get_study_paths(run_dir, orbit_id)

    # Run Sorcha to generate synthetic observations
    observations = run_sorcha(
        impactor_orbit,
        pointing_file,
        paths["sorcha_dir"],
        seed=seed,
    )

    if len(observations) == 0:
        return ImpactStudyResults.empty()

    # Sort the observations by time and origin code
    observations = observations.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos", "coordinates.origin.code"]
    )

    # Add the observing night column to the observations
    observations = observations.set_column(
        "observing_night",
        calculate_observing_night(
            observations.coordinates.origin.code, observations.coordinates.time
        ),
    )

    # Select the unique nights of observations and
    unique_nights = pc.unique(observations.observing_night).sort()

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
        mask = pc.less_equal(observations.observing_night, night)
        observations_window = observations.apply_mask(mask)

        if max_processes == 1:
            result = calculate_impact_probability(
                observations_window,
                impactor_orbit,
                propagator_class,
                run_dir,
                max_processes,
                seed=seed,
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
                    run_dir,
                    max_processes,
                    seed=seed,
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

    # Sort the results by observation_end for consistency.
    results = results.sort_by("observation_end")

    results.to_parquet(f"{paths['object_base_dir']}/impact_results_{orbit_id}.parquet")

    return results


run_impact_study_fo_remote = ray.remote(run_impact_study_fo)


def calculate_impact_probability(
    observations: Observations,
    impactor_orbit: Orbits,
    propagator_class: Type[ASSISTPropagator],
    run_dir: str,
    max_processes: int = 1,
    seed: Optional[int] = None,
) -> ImpactStudyResults:
    """Calculate impact probability for a set of observations.

    Parameters
    ----------
    observations : Observations
        Observations to calculate an orbit from and determine impact probability.
    impactor_orbit : Orbits
        Original impactor orbit
    propagator_class : Type[ASSISTPropagator]
        Propagator class
    run_dir : str
        Directory for this study run
    max_processes : int
        Maximum number of processes to use for impact calculation

    Returns
    -------
    ImpactStudyResults
        Impact probability results for this day if successful
    """
    # if observing_night is null, we need to add it
    if pc.any(pc.is_null(observations.observing_night)).as_py():
        observations = observations.set_column(
            "observing_night",
            calculate_observing_night(
                observations.coordinates.origin.code, observations.coordinates.time
            ),
        )

    orbit_id = impactor_orbit.orbit_id[0].as_py()
    object_id = impactor_orbit.object_id[0].as_py()
    start_night = pc.min(observations.observing_night)
    end_night = pc.max(observations.observing_night)
    start_date = observations.coordinates.time.min()
    end_date = observations.coordinates.time.max()
    window_name = f"{start_night.as_py()}_{end_night.as_py()}"
    paths = get_study_paths(run_dir, orbit_id, window_name)

    thirty_days_before_impact = impactor_orbit.impact_time.add_days(-30)

    # Get the start and end date of the observations, the number of
    # observations, and the number of unique nights
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
            orbit_id=[orbit_id],
            object_id=[object_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            error=[str(e)],
        )

    if error is not None:
        return ImpactStudyResults.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[error],
        )

    try:
        propagator = propagator_class()
        orbit.to_parquet(f"{paths['propagated']}/not_propagated.parquet")
        propagated_30_days_before_impact = propagator.propagate_orbits(
            orbit,
            thirty_days_before_impact,
            covariance=True,
            covariance_method="monte-carlo",
            max_processes=max_processes,
            num_samples=1000,
            seed=seed,
        )
        propagated_30_days_before_impact.to_parquet(
            f"{paths['propagated']}/orbits.parquet"
        )
    except Exception as e:
        logger.error(f"Error propagating orbits: {e}")
        return ImpactStudyResults.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[str(e)],
        )

    try:
        propagator = propagator_class()
        final_orbit_states, impacts = calculate_impacts(
            propagated_30_days_before_impact,
            60,
            propagator,
            num_samples=10000,
            processes=max_processes,
            seed=seed,
        )
        final_orbit_states.to_parquet(
            f"{paths['propagated']}/monte_carlo_variant_states.parquet"
        )

        ip = calculate_impact_probabilities(final_orbit_states, impacts)
    except Exception as e:
        return ImpactStudyResults.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[str(e)],
        )

    return ImpactStudyResults.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[object_id],
        observation_start=start_date,
        observation_end=end_date,
        observation_count=[observations_count],
        observation_nights=[observation_nights],
        observations_rejected=[len(rejected_observations)],
        impact_probability=ip.cumulative_probability,
    )


# Create remote version
calculate_impact_probability_remote = ray.remote(calculate_impact_probability)
