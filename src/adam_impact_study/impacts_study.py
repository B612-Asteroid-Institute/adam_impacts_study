import logging
import os
import shutil
from typing import Iterator, Optional, Type

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_assist import ASSISTPropagator
from adam_core.dynamics.impacts import calculate_impact_probabilities
from adam_core.observations.ades import ADESObservations
from adam_core.observers.utils import calculate_observing_night
from adam_core.orbits import VariantOrbits
from adam_core.time import Timestamp

from adam_impact_study.conversions import Observations
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.sorcha_utils import run_sorcha
from adam_impact_study.types import (
    ImpactorOrbits,
    OrbitWithWindowName,
    VariantOrbitsWithWindowName,
    WindowResult,
)
from adam_impact_study.utils import get_study_paths

from .utils import seed_from_string

logger = logging.getLogger(__name__)

logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))


def run_impact_study_all(
    impactor_orbits: ImpactorOrbits,
    pointing_file: str,
    run_dir: str,
    monte_carlo_samples: int,
    assist_epsilon: float,
    assist_min_dt: float,
    assist_initial_dt: float,
    assist_adaptive_mode: int,
    max_processes: Optional[int] = 1,
    overwrite: bool = True,
    seed: Optional[int] = 13612,
) -> Optional[WindowResult]:
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
            self.initial_dt = assist_initial_dt
            self.min_dt = assist_min_dt
            self.adaptive_mode = assist_adaptive_mode
            self.epsilon = assist_epsilon

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

    impact_results = WindowResult.empty()

    futures = []
    for orbit_id in orbit_ids:
        impactor_orbit = impactor_orbits.select("orbit_id", orbit_id)

        orbit_seed = seed_from_string(orbit_id.as_py(), seed)

        if max_processes == 1:
            impact_result = run_impact_study_for_orbit(
                impactor_orbit,
                ImpactASSISTPropagator,
                pointing_file,
                run_dir,
                monte_carlo_samples,
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
                    monte_carlo_samples,
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


def run_impact_study_for_orbit(
    impactor_orbit: ImpactorOrbits,
    propagator_class: Type[ASSISTPropagator],
    pointing_file: str,
    run_dir: str,
    monte_carlo_samples: int,
    max_processes: Optional[int] = 1,
    seed: Optional[int] = None,
) -> WindowResult:
    """Run an impact study for a single impactor.

    Individual window results are accumulated but saved to their corresponding
    time window directory.

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

    # Serialize the ImpactorOrbit to a file for future analysis use
    impactor_orbit.to_parquet(
        f"{paths['orbit_base_dir']}/impact_orbits_{orbit_id}.parquet"
    )

    # Run Sorcha to generate synthetic observations
    observations = run_sorcha(
        impactor_orbit,
        pointing_file,
        paths["sorcha_dir"],
        seed=seed,
    )

    # Serialize the observations to a file for future analysis use
    observations.to_parquet(f"{paths['sorcha_dir']}/observations_{orbit_id}.parquet")

    if len(observations) == 0:
        return WindowResult.empty()

    # Select the unique nights of observations and
    unique_nights = pc.unique(observations.observing_night).sort()

    if len(unique_nights) < 3:
        # TODO: We might consider returning something else here.
        return WindowResult.empty()

    # Process each time window
    # We iterate through unique nights and filter observations based on
    # to everything below or equal to the current night number
    # We start with a minimum of three unique nights
    futures = []
    results = WindowResult.empty()
    for night in unique_nights[2:]:
        mask = pc.less_equal(observations.observing_night, night)
        observations_window = observations.apply_mask(mask)

        if max_processes == 1:
            result = calculate_window_impact_probability(
                observations_window,
                impactor_orbit,
                propagator_class,
                run_dir,
                monte_carlo_samples,
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
                calculate_window_impact_probability_remote.remote(
                    observations_window,
                    impactor_orbit,
                    propagator_class,
                    run_dir,
                    monte_carlo_samples,
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

    return results


run_impact_study_fo_remote = ray.remote(run_impact_study_for_orbit)


def calculate_window_impact_probability(
    observations: Observations,
    impactor_orbit: ImpactorOrbits,
    propagator_class: Type[ASSISTPropagator],
    run_dir: str,
    monte_carlo_samples: int,
    max_processes: int = 1,
    seed: Optional[int] = None,
) -> WindowResult:
    """Calculate impact probability for a set of observations.

    Parameters
    ----------
    observations : Observations
        Observations to calculate an orbit from and determine impact probability.
    impactor_orbit : ImpactorOrbits
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
    window = f"{start_night.as_py()}_{end_night.as_py()}"
    paths = get_study_paths(run_dir, orbit_id, window)

    # Get the start and end date of the observations, the number of
    # observations, and the number of unique nights
    observations_count = len(observations)
    unique_nights = pc.unique(observations.observing_night).sort()
    num_observation_nights = len(unique_nights)

    rejected_observations = ADESObservations.empty()

    try:
        orbit, rejected_observations, error = run_fo_od(
            observations,
            paths["fo_dir"],
        )

        # Persist the window orbit with the window name for future analysis
        orbit_with_window = OrbitWithWindowName.from_kwargs(
            window=pa.repeat(window, len(orbit)),
            orbit=orbit,
        )
        orbit_with_window.to_parquet(f"{paths['time_dir']}/orbit_with_window.parquet")
    except Exception as e:
        return WindowResult.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            window=[window],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[num_observation_nights],
            error=[str(e)],
        )

    if error is not None:
        return WindowResult.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            window=[window],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[num_observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[error],
        )

    days_until_impact_plus_thirty = (
        int(
            impactor_orbit.impact_time.mjd()[0].as_py()
            - orbit.coordinates.time.mjd()[0].as_py()
        )
        + 30
    )

    try:
        propagator = propagator_class()

        # Create initial variants
        variants = VariantOrbits.create(
            orbit, method="monte-carlo", num_samples=monte_carlo_samples, seed=seed
        )
        variants_with_window = VariantOrbitsWithWindowName.from_kwargs(
            window=pa.repeat(window, len(variants)),
            variant=variants,
        )
        # Persist the initial state of the variants with the window name
        # for future analysis
        variants_with_window.to_parquet(f"{paths['time_dir']}/initial_variants.parquet")

        final_orbit_states, impacts = propagator.detect_impacts(
            variants_with_window.variant,
            days_until_impact_plus_thirty,
            max_processes=max_processes,
        )

        final_orbit_states_with_window = VariantOrbitsWithWindowName.from_kwargs(
            window=pa.repeat(window, len(final_orbit_states)),
            variant=final_orbit_states,
        )

        final_orbit_states_with_window.to_parquet(
            f"{paths['time_dir']}/final_variants.parquet"
        )

        impacts.to_parquet(f"{paths['time_dir']}/impacts.parquet")
        ip = calculate_impact_probabilities(final_orbit_states, impacts)

    except Exception as e:
        return WindowResult.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            window=[window],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[num_observation_nights],
            observations_rejected=[len(rejected_observations)],
            error=[str(e)],
        )

    window_result = WindowResult.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[object_id],
        window=[window],
        observation_start=start_date,
        observation_end=end_date,
        observation_count=[observations_count],
        observation_nights=[num_observation_nights],
        observations_rejected=[len(rejected_observations)],
        impact_probability=ip.cumulative_probability,
        mean_impact_time=ip.mean_impact_time,
        minimum_impact_time=ip.minimum_impact_time,
        maximum_impact_time=ip.maximum_impact_time,
        stddev_impact_time=ip.stddev_impact_time,
    )
    window_result.to_parquet(f"{paths['time_dir']}/window_result.parquet")

    return window_result


# Create remote version
calculate_window_impact_probability_remote = ray.remote(
    calculate_window_impact_probability
)
