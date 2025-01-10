import os

import pyarrow.compute as pc
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.analysis import (
    DiscoveryDates,
    RealizationTimes,
    WarningTimes,
    compute_discovery_dates,
    compute_realization_time,
    compute_warning_time,
    plot_ip_over_time,
)
from adam_impact_study.types import ImpactorOrbits, WindowResult


@pytest.fixture
def impact_study_results():
    orbit_ids = ["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"]
    start_dates = Timestamp.from_mjd(
        [59800.0, 59800.0, 59800.0, 59800.0, 59800.0, 59800.0]
    )
    end_dates = Timestamp.from_mjd(
        [59801.0, 59802.0, 59803.0, 59801.0, 59802.0, 59803.0]
    )
    observation_counts = [10, 20, 30, 10, 20, 30]
    observation_nights = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    observations_rejected = [0, 0, 0, 0, 0, 0]
    impact_probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    impact_result = WindowResult.from_kwargs(
        orbit_id=orbit_ids,
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
        time=Timestamp.from_mjd(
            [59831.0, 59831.0], scale="tdb"
        ),  # 30 days after last observation
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["obj1", "obj2"],
        object_id=["obj1", "obj2"],
        coordinates=cartesian_coords,
    )
    return orbits


def test_plot_ip_over_time(impact_study_results, impacting_orbits, tmpdir):
    # tmpdir_path = tmpdir.mkdir("plots")
    tmpdir_path = os.path.join(os.getcwd(), "test_plots")
    os.makedirs(tmpdir_path, exist_ok=True)

    # Test without survey_start
    plot_ip_over_time(
        impacting_orbits, impact_study_results, tmpdir_path, survey_start=None
    )
    orbit_ids = impact_study_results.orbit_id.unique()
    for obj_id in orbit_ids:
        assert os.path.exists(os.path.join(tmpdir_path, f"{obj_id}/IP_{obj_id}.png"))

    # Test with survey_start
    survey_start = Timestamp.from_mjd(
        [59790.0], scale="utc"
    )  # 10 days before first observation
    plot_ip_over_time(
        impacting_orbits, impact_study_results, tmpdir_path, survey_start=survey_start
    )
    for obj_id in orbit_ids:
        assert os.path.exists(os.path.join(tmpdir_path, f"{obj_id}/IP_{obj_id}.png"))

    print(tmpdir_path)


def test_compute_discovery_dates():
    # Create test impactor orbits
    impactor_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        object_id=["obj1", "obj2", "obj3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1, 1, 1],
            y=[1, 1, 1],
            z=[1, 1, 1],
            vx=[0, 0, 0],
            vy=[0, 0, 0],
            vz=[0, 0, 0],
            time=Timestamp.from_mjd([60000, 60000, 60000]),
        ),
        impact_time=Timestamp.from_mjd([60100, 60200, 60300]),
        dynamical_class=["APO", "APO", "APO"],
        ast_class=["C", "S", "C"],
        diameter=[1.0, 1.0, 1.0],
        albedo=[0.1, 0.1, 0.1],
        H_r=[20.0, 20.0, 20.0],
        u_r=[0.0, 0.0, 0.0],
        g_r=[0.0, 0.0, 0.0],
        i_r=[0.0, 0.0, 0.0],
        z_r=[0.0, 0.0, 0.0],
        y_r=[0.0, 0.0, 0.0],
        GS=[0.15, 0.15, 0.15],
    )

    # Create test results with varying observation nights
    results = WindowResult.from_kwargs(
        orbit_id=["test1", "test1", "test2", "test2", "test3"],
        object_id=["obj1", "obj1", "obj2", "obj2", "obj3"],
        observation_start=Timestamp.from_mjd(
            [60000, 60000, 60000, 60010, 60000], scale="utc"
        ),
        observation_end=Timestamp.from_mjd(
            [60002, 60012, 60002, 60012, 60002], scale="utc"
        ),
        observation_count=[10, 15, 5, 8, 3],
        observations_rejected=[0, 0, 0, 0, 0],
        observation_nights=[2, 4, 1, 2, 1],  # obj1 has 4 nights, obj2 has 2, obj3 has 1
        impact_probability=[0.1, 0.2, 0.1, 0.2, 0.1],
        impact_time=Timestamp.from_mjd(
            [60100, 60100, 60200, 60200, 60300], scale="utc"
        ),
        error=[None, None, None, None, None],
        car_coordinates=None,
        kep_coordinates=None,
    )

    # Compute discovery dates
    discovery_dates = compute_discovery_dates(impactor_orbits, results)

    # Check we got results for all objects
    assert len(discovery_dates) == 3

    # Only obj1 should have a discovery date since it's the only one with >= 3 nights
    obj1_date = discovery_dates.select("orbit_id", "test1").discovery_date[0]
    assert obj1_date is not None
    assert obj1_date.equals(Timestamp.from_mjd([60012], scale="utc")).to_pylist()[
        0
    ]  # Should be the end time of the window with 4 nights

    # obj2 and obj3 should have null discovery dates
    assert pc.all(
        discovery_dates.select("orbit_id", "test2").discovery_date.null_mask()
    )
    assert pc.all(
        discovery_dates.select("orbit_id", "test3").discovery_date.null_mask()
    )


def test_compute_warning_time():
    # Create test impactor orbits
    impactor_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        object_id=["obj1", "obj2", "obj3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1, 1, 1],
            y=[1, 1, 1],
            z=[1, 1, 1],
            vx=[0, 0, 0],
            vy=[0, 0, 0],
            vz=[0, 0, 0],
            time=Timestamp.from_mjd([60000, 60000, 60000]),
        ),
        impact_time=Timestamp.from_mjd([60100, 60200, 60300]),  # Different impact times
        dynamical_class=["APO", "APO", "APO"],
        ast_class=["C", "S", "C"],
        diameter=[1.0, 1.0, 1.0],
        albedo=[0.1, 0.1, 0.1],
        H_r=[20.0, 20.0, 20.0],
        u_r=[0.0, 0.0, 0.0],
        g_r=[0.0, 0.0, 0.0],
        i_r=[0.0, 0.0, 0.0],
        z_r=[0.0, 0.0, 0.0],
        y_r=[0.0, 0.0, 0.0],
        GS=[0.15, 0.15, 0.15],
    )

    # Create test results
    results = WindowResult.from_kwargs(
        orbit_id=["test1", "test1", "test2", "test3"],
        observation_start=Timestamp.from_mjd([60000, 60010, 60000, 60000]),
        observation_end=Timestamp.from_mjd([60050, 60060, 60150, 60250]),
        observation_count=[10, 15, 5, 20],
        observations_rejected=[0, 0, 0, 0],
        observation_nights=[1, 1, 1, 1],
        impact_probability=[
            0.2,
            0.3,
            0.00001,
            0.5,
        ],  # obj1 has two entries, obj2 below threshold
    )

    # Compute warning times
    warning_times = compute_warning_time(impactor_orbits, results, threshold=1e-4)

    assert len(warning_times) == 3

    # Check object IDs are present
    assert pc.any(pc.equal(warning_times.orbit_id, "test1")).as_py()
    assert pc.any(pc.equal(warning_times.orbit_id, "test2")).as_py()
    assert pc.any(pc.equal(warning_times.orbit_id, "test3")).as_py()

    # Check warning time
    warning_time_obj1 = warning_times.select("orbit_id", "test1")
    assert warning_time_obj1.warning_time[0].as_py() == 50.0  # 60100 - 60050

    warning_time_obj2 = warning_times.select("orbit_id", "test2")
    assert pc.all(pc.is_null(warning_time_obj2.warning_time)).as_py()

    warning_time_obj3 = warning_times.select("orbit_id", "test3")
    assert warning_time_obj3.warning_time[0].as_py() == 50.0  # 60300 - 60250


def test_compute_warning_time_edge_cases():
    # Test empty impact study results
    impactor_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        object_id=["obj1", "obj2", "obj3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1, 1, 1],
            y=[1, 1, 1],
            z=[1, 1, 1],
            vx=[0, 0, 0],
            vy=[0, 0, 0],
            vz=[0, 0, 0],
            time=Timestamp.from_mjd([60000, 60000, 60000]),
        ),
        impact_time=Timestamp.from_mjd([60100, 60200, 60300]),  # Different impact times
        dynamical_class=["APO", "APO", "APO"],
        ast_class=["C", "S", "C"],
        diameter=[1.0, 1.0, 1.0],
        albedo=[0.1, 0.1, 0.1],
        H_r=[20.0, 20.0, 20.0],
        u_r=[0.0, 0.0, 0.0],
        g_r=[0.0, 0.0, 0.0],
        i_r=[0.0, 0.0, 0.0],
        z_r=[0.0, 0.0, 0.0],
        y_r=[0.0, 0.0, 0.0],
        GS=[0.15, 0.15, 0.15],
    )
    empty_results = WindowResult.empty()

    empty_warning_times = compute_warning_time(impactor_orbits, empty_results)
    assert len(empty_warning_times) == 3
    assert pc.all(pc.is_null(empty_warning_times.column("warning_time"))).as_py()

    # Test all probabilities below threshold
    low_prob_results = WindowResult.from_kwargs(
        orbit_id=["obj1"],
        observation_start=Timestamp.from_mjd([60000]),
        observation_end=Timestamp.from_mjd([60050]),
        observation_count=[10],
        observations_rejected=[0],
        observation_nights=[1],
        impact_probability=[0.00001],  # Below default threshold
    )

    low_prob_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["test1"],
        object_id=["obj1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1],
            y=[1],
            z=[1],
            vx=[0],
            vy=[0],
            vz=[0],
            time=Timestamp.from_mjd([60000]),
        ),
        impact_time=Timestamp.from_mjd([60100]),
        dynamical_class=["APO"],
        ast_class=["C"],
        diameter=[1.0],
        albedo=[0.1],
        H_r=[20.0],
        u_r=[0.0],
        g_r=[0.0],
        i_r=[0.0],
        z_r=[0.0],
        y_r=[0.0],
        GS=[0.15],
    )

    low_prob_warning_times = compute_warning_time(low_prob_orbits, low_prob_results)
    assert len(low_prob_warning_times) == 1


def test_compute_realization_time():
    # Create test impactor orbits
    impactor_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        object_id=["obj1", "obj2", "obj3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1, 1, 1],
            y=[1, 1, 1],
            z=[1, 1, 1],
            vx=[0, 0, 0],
            vy=[0, 0, 0],
            vz=[0, 0, 0],
            time=Timestamp.from_mjd([60000, 60000, 60000]),
        ),
        impact_time=Timestamp.from_mjd([60500, 60300, 60350]),  # Different impact times
        dynamical_class=["APO", "APO", "APO"],
        ast_class=["C", "S", "C"],
        diameter=[1.0, 1.0, 1.0],
        albedo=[0.1, 0.1, 0.1],
        H_r=[20.0, 20.0, 20.0],
        u_r=[0.0, 0.0, 0.0],
        g_r=[0.0, 0.0, 0.0],
        i_r=[0.0, 0.0, 0.0],
        z_r=[0.0, 0.0, 0.0],
        y_r=[0.0, 0.0, 0.0],
        GS=[0.15, 0.15, 0.15],
    )

    # Create test results
    results = WindowResult.from_kwargs(
        orbit_id=["test1", "test1", "test2", "test3"],
        observation_start=Timestamp.from_mjd([60000, 60010, 60000, 60000]),
        observation_end=Timestamp.from_mjd([60050, 60075, 60150, 60250]),
        observation_count=[10, 15, 5, 20],
        observations_rejected=[0, 0, 0, 0],
        observation_nights=[1, 1, 1, 1],
        impact_probability=[
            0.0,
            0.3,
            1e-10,
            0.5,
        ],  # obj1 has two entries, obj2 below threshold
    )

    discovery_dates = DiscoveryDates.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        discovery_date=Timestamp.from_kwargs(
            days=[60050, None, 60250],
            nanos=[0, None, 0],
            scale="utc",
            permit_nulls=True,
        ),
    )

    realization_times = compute_realization_time(
        impactor_orbits, results, discovery_dates
    )

    assert len(realization_times) == 3
    assert realization_times.orbit_id.to_pylist() == ["test1", "test2", "test3"]
    assert realization_times.realization_time.to_pylist() == [25.0, None, 0.0]

    realization_times = compute_realization_time(
        impactor_orbits, results, discovery_dates, threshold=0.4
    )

    assert len(realization_times) == 3
    assert realization_times.orbit_id.to_pylist() == ["test1", "test2", "test3"]
    assert realization_times.realization_time.to_pylist() == [None, None, 0.0]
