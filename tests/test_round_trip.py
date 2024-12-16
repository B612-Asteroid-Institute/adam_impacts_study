import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pytest
import quivr as qv
import ray
from adam_assist import ASSISTPropagator
from adam_core.constants import KM_P_AU
from adam_core.orbits import Orbits
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp


@pytest.fixture
def non_impacting_orbit_fixture():
    return non_impacting_orbit()

def non_impacting_orbits():
    """Load all non-impacting orbits from the round trip data."""
    orbit_path = os.path.join(os.path.dirname(__file__), "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_non_impacting_objects.parquet")
    return Orbits.from_parquet(orbit_path)

def non_impacting_orbit():
    """Load a single non-impacting orbit from the round trip data."""
    orbits = non_impacting_orbits()
    return orbits[1]

def impacting_orbit():
    """Load a single impacting orbit from the round trip data."""
    orbit_path = os.path.join(os.path.dirname(__file__), "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_impacting_objects.parquet")
    orbits = Orbits.from_parquet(orbit_path)
    # Take just the first orbit 
    return orbits[0:1]

@pytest.fixture
def impacting_orbit_fixture():
    return impacting_orbit()


def test_round_trip_propagation_impacting(impacting_orbit_fixture):
    propagator = ASSISTPropagator()
    time_2025 = Timestamp.from_iso8601("2025-05-05T00:00:00")
    original_epoch = impacting_orbit_fixture.coordinates.time
    forward_orbit = propagator.propagate_orbits(impacting_orbit_fixture, time_2025)
    back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch)
    

    diff = impacting_orbit_fixture.coordinates.r - back_orbit.coordinates.r
    norm_diff = np.linalg.norm(diff)
    # Convert diff from au to km
    diff_km = norm_diff * KM_P_AU
    results, impacts = propagator.detect_impacts(back_orbit, 60)
    
    print(f"Normalized difference: {diff_km} km")

    assert len(impacts) == 1



class RoundTripResults(qv.Table):
    orbit_id = qv.LargeStringColumn(nullable=True)
    initial_dt = qv.Float64Column(nullable=True)
    min_dt = qv.Float64Column(nullable=True)
    adaptive_mode = qv.Int64Column(nullable=True)
    position_diff_km = qv.Float64Column(nullable=True)
    num_impacts = qv.Int64Column(nullable=True)
    forward_steps_done = qv.Int64Column(nullable=True)
    backward_steps_done = qv.Int64Column(nullable=True)

@ray.remote
def round_trip_worker(orbit: Orbits, min_dt: float, initial_dt: float, adaptive_mode: int):
    propagator = ASSISTPropagator(min_dt=min_dt, initial_dt=initial_dt, adaptive_mode=adaptive_mode)
    time_2025 = Timestamp.from_iso8601("2025-05-05T00:00:00")
    original_epoch = orbit.coordinates.time
    forward_orbit = propagator.propagate_orbits(orbit, time_2025)
    forward_steps_done = propagator._last_simulation.steps_done
    back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch)
    backward_steps_done = propagator._last_simulation.steps_done
    diff = orbit.coordinates.r - back_orbit.coordinates.r
    norm_diff = np.linalg.norm(diff)
    diff_km = norm_diff * KM_P_AU

    # Check for impacts
    impact_results, impacts = propagator.detect_impacts(back_orbit, 60)
    
    return RoundTripResults.from_kwargs(
        orbit_id=orbit.orbit_id,
        initial_dt=[initial_dt],
        min_dt=[min_dt],
        adaptive_mode=[adaptive_mode],
        position_diff_km=[diff_km],
        num_impacts=[len(impacts)],
        forward_steps_done=[forward_steps_done],
        backward_steps_done=[backward_steps_done]
    )


def test_round_trip_propagation_non_impacting(orbits: Orbits):
    """Run round trip propagation tests with various parameters and collect results."""

    # min_dts = [1e-15, 1e-16, 1e-18, 1e-20, 1e-24]
    min_dts = np.logspace(1, -15, 17)
    # initial_dts = [1e-3, 1e-6, 1e-12, 1e-16, 1e-24]
    initial_dts = np.logspace(1, -12, 14)
    adaptive_modes = [2]
    
    initialize_use_ray(mp.cpu_count())
    futures = []
    all_results = RoundTripResults.empty()
    for adaptive_mode in adaptive_modes:
        for min_dt in min_dts:
            for initial_dt in initial_dts:
                if min_dt > initial_dt:
                    continue
                
                for orbit in orbits:
                    futures.append(round_trip_worker.remote(orbit, min_dt, initial_dt, adaptive_mode))

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        all_results = qv.concatenate([all_results, result])


    return all_results