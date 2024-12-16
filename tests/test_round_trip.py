import os

import numpy as np
import pytest
from adam_assist import ASSISTPropagator
from adam_core.orbits import Orbits
from adam_core.time import Timestamp


@pytest.fixture
def non_impacting_orbit():
    """Load a single non-impacting orbit from the round trip data."""
    orbit_path = os.path.join(os.path.dirname(__file__), "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_non_impacting_objects.parquet")
    orbits = Orbits.from_parquet(orbit_path)
    # Take just the first orbit 
    return orbits[0:1]

@pytest.fixture
def impacting_orbit():
    """Load a single impacting orbit from the round trip data."""
    orbit_path = os.path.join(os.path.dirname(__file__), "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_impacting_objects.parquet")
    orbits = Orbits.from_parquet(orbit_path)
    # Take just the first orbit 
    return orbits[0:1]

def test_round_trip_propagation_impacting(impacting_orbit):
    propagator = ASSISTPropagator()
    time_2025 = Timestamp.from_iso8601("2025-05-05T00:00:00")
    original_epoch = impacting_orbit.coordinates.time
    forward_orbit = propagator.propagate_orbits(impacting_orbit, time_2025)
    back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch)
    
    # Calculate normalized distance between original and round-tripped state vectors
    original_state = impacting_orbit.coordinates.values
    round_trip_state = back_orbit.coordinates.values
    
    diff = impacting_orbit.coordinates.r - back_orbit.coordinates.r
    norm_diff = np.linalg.norm(diff)
    # Convert diff from au to km
    diff_km = norm_diff * 1.495978707e8
    results, impacts = propagator.detect_impacts(back_orbit, 60)
    
    print(f"Normalized difference: {diff_km} km")
    
    # assert norm_diff < 1e-6
    assert len(impacts) == 1

def test_round_trip_propagation_non_impacting(non_impacting_orbit):
    # Initialize propagator
    propagator = ASSISTPropagator()
    
    # Define the time points
    time_2025 = Timestamp.from_iso8601("2025-05-05T00:00:00")
    original_epoch = non_impacting_orbit.coordinates.time
    
    # Back propagation to 2025
    forward_orbit = propagator.propagate_orbits(non_impacting_orbit, time_2025)
    
    # Now forward propagation to 2125
    back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch)
    
    # Calculate normalized distance between original and round-tripped state vectors
    original_state = non_impacting_orbit.coordinates.values
    round_trip_state = back_orbit.coordinates.values
    
    diff = original_state - round_trip_state
    norm_diff = np.linalg.norm(diff) / np.linalg.norm(original_state)
    

    print(f"\nOriginal state vector: {original_state}")
    print(f"Round-tripped state vector: {round_trip_state}")
    print(f"Normalized difference: {norm_diff}")
    
    # Check for impacts
    results, impacts = propagator.detect_impacts(back_orbit, 60)
    
    print(f"Number of impacts detected: {len(impacts)}")
    if len(impacts) > 0:
        print(f"Impact times: {impacts.coordinates.time.mjd()}")
    
    # Assert that the normalized difference is small (you may need to adjust this threshold)
    assert norm_diff < 1e-6
    assert len(impacts) == 1