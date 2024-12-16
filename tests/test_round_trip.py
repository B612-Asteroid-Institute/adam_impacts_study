import os

import numpy as np
import pandas as pd
import pytest
from adam_assist import ASSISTPropagator
from adam_core.constants import KM_P_AU
from adam_core.orbits import Orbits
from adam_core.time import Timestamp


@pytest.fixture
def non_impacting_orbit_fixture():
    return non_impacting_orbit()

def non_impacting_orbit():
    """Load a single non-impacting orbit from the round trip data."""
    orbit_path = os.path.join(os.path.dirname(__file__), "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_non_impacting_objects.parquet")
    orbits = Orbits.from_parquet(orbit_path)
    # Take just the first orbit 
    return orbits[0:1]

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



def test_round_trip_propagation_non_impacting(non_impacting_orbit_fixture):
    """Run round trip propagation tests with various parameters and collect results."""
    all_results = []
    
    # min_dts = [1e-15, 1e-16, 1e-18, 1e-20, 1e-24]
    min_dts = np.logspace(-12, -24, 13)
    # initial_dts = [1e-3, 1e-6, 1e-12, 1e-16, 1e-24]
    initial_dts = np.logspace(-3, -24, 22)
    adaptive_modes = [2]
    
    for adaptive_mode in adaptive_modes:
        for min_dt in min_dts:
            for initial_dt in initial_dts:
                if min_dt > initial_dt:
                    continue
                
                # Initialize propagator
                propagator = ASSISTPropagator(
                    min_dt=min_dt,
                    initial_dt=initial_dt,
                    adaptive_mode=adaptive_mode
                )
                
                # Define the time points
                time_2025 = Timestamp.from_iso8601("2025-05-05T00:00:00")
                original_epoch = non_impacting_orbit_fixture.coordinates.time
                
                # Back propagation to 2025
                forward_orbit = propagator.propagate_orbits(non_impacting_orbit_fixture, time_2025)
                
                # Now forward propagation to 2125
                back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch)
                
                diff = non_impacting_orbit_fixture.coordinates.r - back_orbit.coordinates.r
                norm_diff = np.linalg.norm(diff)
                diff_km = norm_diff * KM_P_AU
                
                # Check for impacts
                impact_results, impacts = propagator.detect_impacts(back_orbit, 60)
                
                # Store results
                all_results.append({
                    'initial_dt': initial_dt,
                    'min_dt': min_dt,
                    'adaptive_mode': adaptive_mode,
                    'position_diff_km': diff_km,
                    'num_impacts': len(impacts)
                })
    
    # Create DataFrame and display results
    df = pd.DataFrame(all_results)
    df = df.sort_values(['initial_dt', 'min_dt'])
    print("\nTest Results:\n")
    print(df.to_string(index=False))

    return df