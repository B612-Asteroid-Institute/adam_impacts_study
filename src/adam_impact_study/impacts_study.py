import logging
import os
import shutil
import time
from typing import Iterator, Optional, Tuple, Type

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_assist import ASSISTPropagator
from adam_core.constants import KM_P_AU
from adam_core.constants import Constants as c
from adam_core.coordinates import Origin
from adam_core.dynamics.impacts import (
    CollisionConditions,
    calculate_impact_probabilities,
)
from adam_core.observations.ades import ADESObservations
from adam_core.observers.utils import calculate_observing_night
from adam_core.orbits import VariantOrbits
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_fo.config import check_build_exists

from adam_impact_study.conversions import Observations
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.sorcha_utils import run_sorcha
from adam_impact_study.types import (
    ImpactorOrbits,
    OrbitWithWindowName,
    ResultsTiming,
    VariantOrbitsWithWindowName,
    WindowResult,
)
from adam_impact_study.utils import get_study_paths

from .utils import seed_from_string

logger = logging.getLogger(__name__)

logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))


EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU


def run_impact_study_all(
    impactor_orbits: ImpactorOrbits,
    pointing_file: str,
    run_dir: str,
    monte_carlo_samples: int,
    assist_epsilon: float,
    assist_min_dt: float,
    assist_initial_dt: float,
    assist_adaptive_mode: int,
    conditions: Optional[CollisionConditions] = None,
    max_processes: Optional[int] = 1,
    overwrite: bool = False,
    seed: Optional[int] = 13612,
) -> Tuple[WindowResult, ResultsTiming]:
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
    # Test that the build fo script has been run
    check_build_exists()

    class ImpactASSISTPropagator(ASSISTPropagator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_dt = assist_initial_dt
            self.min_dt = assist_min_dt
            self.adaptive_mode = assist_adaptive_mode
            self.epsilon = assist_epsilon

    # If the run directory already exists, throw an exception
    # unless the user has specified the overwrite flag
    if os.path.exists(run_dir):
        if overwrite:
            logger.warning(f"Overwriting run directory {run_dir}")
            shutil.rmtree(run_dir)
        else:
            logger.warning(
                f"Run directory {run_dir} already exists, attempting to continue previous run..."
            )

    if conditions is None:
        logger.info(
            "No collision conditions provided, using default Earth impact conditions"
        )
        conditions = CollisionConditions.from_kwargs(
            condition_id=["Default - Earth"],
            collision_object=Origin.from_kwargs(code=["EARTH"]),
            collision_distance=[EARTH_RADIUS_KM],
            stopping_condition=[True],
        )

    os.makedirs(run_dir, exist_ok=True)

    # Initialize ray cluster
    use_ray = initialize_use_ray(num_cpus=max_processes)

    logger.info(f"Impactor Orbits: {impactor_orbits}")
    orbit_ids = impactor_orbits.orbit_id.unique()

    impact_results = WindowResult.empty()
    results_timings = ResultsTiming.empty()
    futures = []
    # If we are only running a single orbit on this machine,
    # we can run the sub-remotes using max_processes
    # Otherwise, each orbit gets a single CPU to run.
    sub_remote_max_processes = 1
    if len(orbit_ids) == 1:
        sub_remote_max_processes = max_processes

    for orbit_id in orbit_ids:
        impactor_orbit = impactor_orbits.select("orbit_id", orbit_id)

        orbit_seed = seed_from_string(orbit_id.as_py(), seed)

        if not use_ray:
            impact_result, results_timing = run_impact_study_for_orbit(
                impactor_orbit,
                ImpactASSISTPropagator,
                pointing_file,
                run_dir,
                monte_carlo_samples,
                assist_epsilon,
                assist_min_dt,
                assist_initial_dt,
                assist_adaptive_mode,
                conditions=conditions,
                max_processes=1,
                seed=orbit_seed,
            )
            impact_results = qv.concatenate([impact_results, impact_result])
            results_timings = qv.concatenate([results_timings, results_timing])
        else:

            # run_impact_study_for_orbit has the ability to call other
            # remote functions so when already running in a ray cluster, we
            # want to explicity set max_processes to 1
            futures.append(
                run_impact_study_for_orbit_remote.remote(
                    impactor_orbit,
                    ImpactASSISTPropagator,
                    pointing_file,
                    run_dir,
                    monte_carlo_samples,
                    assist_epsilon,
                    assist_min_dt,
                    assist_initial_dt,
                    assist_adaptive_mode,
                    conditions=conditions,
                    max_processes=sub_remote_max_processes,
                    seed=orbit_seed,
                )
            )

            if len(futures) > max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result, timing = ray.get(finished[0])
                impact_results = qv.concatenate([impact_results, result])
                results_timings = qv.concatenate([results_timings, timing])

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result, timing = ray.get(finished[0])
        impact_results = qv.concatenate([impact_results, result])
        results_timings = qv.concatenate([results_timings, timing])

    return impact_results, results_timings


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
    assist_epsilon: float,
    assist_min_dt: float,
    assist_initial_dt: float,
    assist_adaptive_mode: int,
    conditions: Optional[CollisionConditions] = None,
    max_processes: Optional[int] = 1,
    seed: Optional[int] = None,
) -> Tuple[WindowResult, ResultsTiming]:
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
    monte_carlo_samples : int
        Number of monte carlo samples to use for impact calculation
    assist_epsilon : float
        Epsilon value for ASSIST
    assist_min_dt : float
        Minimum time step for ASSIST
    assist_initial_dt : float
        Initial time step for ASSIST
    assist_adaptive_mode : int
        Adaptive mode for ASSIST
    max_processes : Optional[int]
        Maximum number of processes to use for impact calculation

    Returns
    -------
    ImpactStudyResults
        Table containing the results of the impact study
    """
    orbit_start_time = time.perf_counter()
    assert len(impactor_orbit) == 1, "Only one object supported at a time"
    orbit_id = impactor_orbit.orbit_id[0].as_py()

    paths = get_study_paths(run_dir, orbit_id)

    # Serialize the ImpactorOrbit to a file for future analysis use
    impactor_orbit_file = f"{paths['orbit_base_dir']}/impactor_orbit.parquet"
    if not os.path.exists(impactor_orbit_file):
        impactor_orbit.to_parquet(impactor_orbit_file)
    else:
        impactor_orbit_saved = ImpactorOrbits.from_parquet(impactor_orbit_file)
        impactor_orbit_table = impactor_orbit.flattened_table().drop_columns(
            ["coordinates.covariance.values"]
        )
        impactor_orbit_saved_table = (
            impactor_orbit_saved.flattened_table().drop_columns(
                ["coordinates.covariance.values"]
            )
        )
        assert impactor_orbit_table.equals(
            impactor_orbit_saved_table
        ), "ImpactorOrbit does not match saved version"

    timing_file = f"{paths['orbit_base_dir']}/timings.parquet"
    if os.path.exists(timing_file):
        timings = ResultsTiming.from_parquet(timing_file)
    else:
        timings = ResultsTiming.from_kwargs(
            orbit_id=[orbit_id],
        )

    if conditions is None:
        logger.info(
            "No collision conditions provided, using default Earth impact conditions"
        )
        conditions = CollisionConditions.from_kwargs(
            condition_id=["Default - Earth"],
            collision_object=Origin.from_kwargs(code=["EARTH"]),
            collision_distance=[EARTH_RADIUS_KM],
            stopping_condition=[True],
        )

    observations_file = f"{paths['sorcha_dir']}/observations_{orbit_id}.parquet"
    sorcha_runtime = None
    if not os.path.exists(observations_file):
        # Run Sorcha to generate synthetic observations
        sorcha_start_time = time.perf_counter()
        observations = run_sorcha(
            impactor_orbit,
            # We need to avoid propagation in sorcha past the impact time
            # to avoid weird edge cases where the propagation gets ejected
            # from the solar system / galaxy, etc...
            impactor_orbit.impact_time.add_days(-1),
            pointing_file,
            paths["sorcha_dir"],
            assist_epsilon=assist_epsilon,
            assist_min_dt=assist_min_dt,
            assist_initial_dt=assist_initial_dt,
            assist_adaptive_mode=assist_adaptive_mode,
            seed=seed,
        )
        sorcha_runtime = time.perf_counter() - sorcha_start_time
        # Serialize the observations to a file for future analysis use
        observations.to_parquet(
            f"{paths['sorcha_dir']}/observations_{orbit_id}.parquet"
        )

        # Update timings
        timings = timings.set_column("sorcha_runtime", pa.array([sorcha_runtime]))
        timings.to_parquet(timing_file)
    else:
        observations = Observations.from_parquet(observations_file)
        logger.info(f"Loaded observations from {observations_file}")

    if len(observations) == 0:
        return WindowResult.empty(), ResultsTiming.empty()

    # Select the unique nights of observations and
    unique_nights = pc.unique(observations.observing_night).sort()

    if len(unique_nights) < 3:
        # TODO: We might consider returning something else here.
        return WindowResult.empty(), ResultsTiming.empty()

    # Initialize ray cluster
    use_ray = initialize_use_ray(num_cpus=max_processes)

    # Process each time window
    # We iterate through unique nights and filter observations based on
    # to everything below or equal to the current night number
    # We start with a minimum of three unique nights
    futures = []
    results = WindowResult.empty()
    for night in unique_nights[2:]:
        mask = pc.less_equal(observations.observing_night, night)
        observations_window = observations.apply_mask(mask)

        if len(observations_window) < 6:
            logger.warning(
                f"Not enough observations for a least-squares fit for nights up to {night}"
            )
            continue

        if not use_ray:
            result = calculate_window_impact_probability(
                observations_window,
                impactor_orbit,
                propagator_class,
                run_dir,
                monte_carlo_samples,
                conditions,
                max_processes=max_processes,
                seed=seed,
            )
            # Log if any error is present
            if pc.any(pc.invert(pc.is_null(result.error))).as_py():
                logger.warning(f"Error: {result.error}")
            results = qv.concatenate([results, result])

        else:
            futures.append(
                calculate_window_impact_probability_remote.remote(
                    observations_window,
                    impactor_orbit,
                    propagator_class,
                    run_dir,
                    monte_carlo_samples,
                    conditions,
                    max_processes=max_processes,
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

    # Update timings
    orbit_end_time = time.perf_counter()
    total_runtime = orbit_end_time - orbit_start_time
    timings = ResultsTiming.from_kwargs(
        orbit_id=timings.orbit_id,
        total_runtime=[total_runtime],
        sorcha_runtime=timings.sorcha_runtime,
        mean_od_runtime=[pc.mean(results.od_runtime)],
        total_od_runtime=[pc.sum(results.od_runtime)],
        mean_ip_runtime=[pc.mean(results.ip_runtime)],
        total_ip_runtime=[pc.sum(results.ip_runtime)],
        mean_window_runtime=[pc.mean(results.window_runtime)],
        total_window_runtime=[pc.sum(results.window_runtime)],
    )
    timings.to_parquet(timing_file)

    return results, timings


run_impact_study_for_orbit_remote = ray.remote(run_impact_study_for_orbit)


def calculate_window_impact_probability(
    observations: Observations,
    impactor_orbit: ImpactorOrbits,
    propagator_class: Type[ASSISTPropagator],
    run_dir: str,
    monte_carlo_samples: int,
    conditions: CollisionConditions,
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
    monte_carlo_samples : int
        Number of monte carlo samples to use for impact calculation
    conditions : CollisionConditions
        Collision conditions to use for impact calculation
    max_processes : int
        Maximum number of processes to use for impact calculation

    Returns
    -------
    ImpactStudyResults
        Impact probability results for this day if successful
    """
    window_start_time = time.perf_counter()
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

    window_out_file = f"{paths['time_dir']}/window_result.parquet"
    if os.path.exists(window_out_file):
        return WindowResult.from_parquet(window_out_file)

    # Get the start and end date of the observations, the number of
    # observations, and the number of unique nights
    observations_count = len(observations)
    unique_nights = pc.unique(observations.observing_night).sort()
    num_observation_nights = len(unique_nights)

    rejected_observations = ADESObservations.empty()

    od_runtime = None
    try:
        od_start_time = time.perf_counter()
        orbit, rejected_observations, error = run_fo_od(
            observations,
            paths["fo_dir"],
        )
        od_runtime = time.perf_counter() - od_start_time
        # Persist the window orbit with the window name for future analysis
        orbit_with_window = OrbitWithWindowName.from_kwargs(
            window=pa.repeat(window, len(orbit)),
            orbit=orbit,
        )
        orbit_with_window.to_parquet(f"{paths['time_dir']}/orbit_with_window.parquet")
    except Exception as e:
        window_result = WindowResult.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            window=[window],
            status=["failed"],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[num_observation_nights],
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            minimum_impact_time=Timestamp.nulls(1, scale="tdb"),
            maximum_impact_time=Timestamp.nulls(1, scale="tdb"),
            error=[str(e)],
            od_runtime=[od_runtime],
        )
        window_result.to_parquet(window_out_file)
        return window_result

    if error is not None:
        window_result = WindowResult.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            window=[window],
            status=["failed"],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[num_observation_nights],
            observations_rejected=[len(rejected_observations)],
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            minimum_impact_time=Timestamp.nulls(1, scale="tdb"),
            maximum_impact_time=Timestamp.nulls(1, scale="tdb"),
            error=[error],
            od_runtime=[od_runtime],
        )
        window_result.to_parquet(window_out_file)
        return window_result

    days_until_impact_plus_thirty = (
        int(
            impactor_orbit.impact_time.mjd()[0].as_py()
            - orbit.coordinates.time.mjd()[0].as_py()
        )
        + 30
    )

    ip_runtime = None
    try:
        ip_start_time = time.perf_counter()
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

        final_orbit_states, impacts = propagator.detect_collisions(
            variants_with_window.variant,
            days_until_impact_plus_thirty,
            conditions=conditions,
            max_processes=max_processes,
        )
        if len(impacts.condition_id.unique()) > 1:
            logger.warning(
                f"Multiple collision conditions detected: {impacts.condition_id.unique()}"
            )

        final_orbit_states_with_window = VariantOrbitsWithWindowName.from_kwargs(
            window=pa.repeat(window, len(final_orbit_states)),
            variant=final_orbit_states,
        )

        final_orbit_states_with_window.to_parquet(
            f"{paths['time_dir']}/final_variants.parquet"
        )

        impacts.to_parquet(f"{paths['time_dir']}/impacts.parquet")
        ip = calculate_impact_probabilities(
            final_orbit_states, impacts, conditions=conditions
        )
        ip_runtime = time.perf_counter() - ip_start_time

    except Exception as e:
        window_result = WindowResult.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            window=[window],
            status=["failed"],
            observation_start=start_date,
            observation_end=end_date,
            observation_count=[observations_count],
            observation_nights=[num_observation_nights],
            observations_rejected=[len(rejected_observations)],
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            minimum_impact_time=Timestamp.nulls(1, scale="tdb"),
            maximum_impact_time=Timestamp.nulls(1, scale="tdb"),
            error=[str(e)],
            od_runtime=[od_runtime],
            ip_runtime=[ip_runtime],
        )
        window_result.to_parquet(window_out_file)
        return window_result

    print(ip.mean_impact_time.scale)

    window_end_time = time.perf_counter()

    window_result = WindowResult.from_kwargs(
        orbit_id=pa.repeat(orbit_id, len(ip)),
        object_id=pa.repeat(object_id, len(ip)),
        window=pa.repeat(window, len(ip)),
        condition_id=ip.condition_id,
        status=pa.repeat("complete", len(ip)),
        observation_start=start_date.take([0 for _ in range(len(ip))]),
        observation_end=end_date.take([0 for _ in range(len(ip))]),
        observation_count=pa.repeat(observations_count, len(ip)),
        observation_nights=pa.repeat(num_observation_nights, len(ip)),
        observations_rejected=pa.repeat(len(rejected_observations), len(ip)),
        impact_probability=ip.cumulative_probability,
        mean_impact_time=ip.mean_impact_time,
        minimum_impact_time=ip.minimum_impact_time,
        maximum_impact_time=ip.maximum_impact_time,
        stddev_impact_time=ip.stddev_impact_time,
        window_runtime=pa.repeat(window_end_time - window_start_time, len(ip)),
        od_runtime=pa.repeat(od_runtime, len(ip)),
        ip_runtime=pa.repeat(ip_runtime, len(ip)),
    )
    window_result.to_parquet(f"{paths['time_dir']}/window_result.parquet")

    return window_result


# Create remote version
calculate_window_impact_probability_remote = ray.remote(
    calculate_window_impact_probability
)
