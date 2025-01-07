import os

import pytest
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.types import ImpactStudyResults


@pytest.fixture
def impact_study_results():
    object_ids = ["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"]
    start_dates = Timestamp.from_mjd([59800.0, 59800.0, 59800.0, 59800.0, 59800.0, 59800.0])
    end_dates = Timestamp.from_mjd([59801.0, 59802.0, 59803.0, 59801.0, 59802.0, 59803.0])
    observation_counts = [10, 20, 30, 10, 20, 30]
    observation_nights = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    observations_rejected = [0, 0, 0, 0, 0, 0]
    impact_probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    impact_result = ImpactStudyResults.from_kwargs(
        object_id=object_ids,
        observation_start=start_dates,
        observation_end=end_dates,
        observation_count=observation_counts,
        observation_nights=observation_nights,
        impact_probability=impact_probabilities,
        observations_rejected=observations_rejected,
    )

    return impact_result


@pytest.fixture
def impacting_orbits():
    # Create impacting orbits for obj1 and obj2
    cartesian_coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 2.0],
        y=[0.5, 1.5],
        z=[0.1, 0.2],
        vx=[0.01, 0.02],
        vy=[0.005, 0.015],
        vz=[0.001, 0.002],
        time=Timestamp.from_mjd([59831.0, 59831.0], scale="tdb"),  # 30 days after last observation
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["obj1", "obj2"],
        object_id=["obj1", "obj2"],
        coordinates=cartesian_coords
    )
    return orbits


def test_plot_ip_over_time(impact_study_results, impacting_orbits, tmpdir):
    tmpdir_path = tmpdir.mkdir("plots")
    os.chdir(tmpdir_path)
    
    # Test without survey_start
    plot_ip_over_time(impact_study_results, tmpdir_path, impacting_orbits)
    object_ids = impact_study_results.object_id.unique()
    for obj_id in object_ids:
        assert os.path.exists(os.path.join(tmpdir_path, f"IP_{obj_id}.png"))
    
    # Test with survey_start
    survey_start = Timestamp.from_mjd(59790.0, scale="utc")  # 10 days before first observation
    plot_ip_over_time(impact_study_results, tmpdir_path, impacting_orbits, survey_start)
    print(f"Files in directory: {os.listdir(tmpdir_path)}")
    # open the image
    for obj_id in object_ids:
        assert os.path.exists(os.path.join(tmpdir_path, f"IP_{obj_id}.png"))
