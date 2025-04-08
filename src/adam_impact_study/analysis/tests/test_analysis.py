import pyarrow.compute as pc
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.time import Timestamp

from adam_impact_study.analysis import (
    compute_discovery_dates,
    compute_ip_threshold_date,
    compute_realization_time,
    compute_warning_time,
)
from adam_impact_study.analysis.plots import plot_individual_orbit_ip_over_time
from adam_impact_study.types import (
    DiscoveryDates,
    ImpactorOrbits,
    Observations,
    WindowResult,
)


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
    windows = [
        "60000_60002",
        "60000_60012",
        "60000_60002",
        "60000_60012",
        "60000_60002",
        "60000_60012",
    ]
    condition_ids = ["Default - Earth"] * 6
    statuses = ["complete"] * 6
    impact_result = WindowResult.from_kwargs(
        orbit_id=orbit_ids,
        condition_id=condition_ids,
        status=statuses,
        window=windows,
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
    orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["obj1", "obj2"],
        object_id=["obj1", "obj2"],
        coordinates=cartesian_coords,
        impact_time=Timestamp.from_mjd([60100, 60200], scale="utc"),
        dynamical_class=["APO", "APO"],
        ast_class=["C", "S"],
        diameter=[1.0, 1.0],
        albedo=[0.1, 0.1],
        H_r=[20.0, 20.0],
        i_r=[0.0, 0.0],
        u_r=[0.0, 0.0],
        g_r=[0.0, 0.0],
        z_r=[0.0, 0.0],
        y_r=[0.0, 0.0],
        GS=[0.15, 0.15],
    )
    return orbits


def test_plot_ip_over_time(impact_study_results, impacting_orbits, tmpdir):
    tmpdir_path = tmpdir.mkdir("plots")

    # Test without survey_start
    plot_individual_orbit_ip_over_time(
        impacting_orbits, impact_study_results, tmpdir_path, survey_start=None
    )
    orbit_ids = impact_study_results.orbit_id.unique()
    for obj_id in orbit_ids:
        plot_path = tmpdir_path / f"IP_{obj_id}.png"
        assert plot_path.exists()

    other_tmpdir_path = tmpdir.mkdir("other_plots")

    # Test with survey_start
    survey_start = Timestamp.from_mjd(
        [59790.0], scale="utc"
    )  # 10 days before first observation
    plot_individual_orbit_ip_over_time(
        impacting_orbits,
        impact_study_results,
        other_tmpdir_path,
        survey_start=survey_start,
    )
    for obj_id in orbit_ids:
        plot_path = other_tmpdir_path / f"IP_{obj_id}.png"
        assert plot_path.exists()


def test_compute_discovery_dates():

    # Create test results with varying observation nights
    observations = Observations.from_kwargs(
        orbit_id=["test1", "test1", "test1", "test1", "test1", "test1"],
        obs_id=["obs1", "obs1", "obs1", "obs1", "obs1", "obs1"],
        observing_night=[60001, 60001, 60002, 60002, 60003, 60003],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=[180.0, 181.0, 182.0, 183.0, 184.0, 185.0],
            lat=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            time=Timestamp.from_mjd(
                [60001.1, 60001.2, 60002.1, 60002.2, 60003.1, 60003.2]
            ),
            origin=Origin.from_kwargs(code=["X05", "X05", "X05", "X05", "X05", "X05"]),
            frame="equatorial",
        ),
    )

    # first test min_tracklets is working
    min_tracklets = 5
    discovery_dates = compute_discovery_dates(observations, min_tracklets=min_tracklets)
    assert pc.all(pc.is_null(discovery_dates.discovery_date.days)).as_py()
    assert len(discovery_dates) == 1

    # now test max_nights is working
    max_nights = 1
    discovery_dates = compute_discovery_dates(observations, max_nights=max_nights)
    assert pc.all(pc.is_null(discovery_dates.discovery_date.days)).as_py()
    assert len(discovery_dates) == 1

    # now run for real
    discovery_dates = compute_discovery_dates(observations)
    assert discovery_dates.discovery_date.mjd().to_pylist() == [60003.2]
    assert len(discovery_dates) == 1

    # now run for real
    max_nights = 2
    min_tracklets = 2
    discovery_dates = compute_discovery_dates(
        observations, max_nights=max_nights, min_tracklets=min_tracklets
    )
    assert discovery_dates.discovery_date.mjd().to_pylist() == [60002.2]
    assert len(discovery_dates) == 1


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
        impact_time=Timestamp.from_mjd([60100, 60200, 60350]),  # Different impact times
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

    discovery_dates = DiscoveryDates.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        discovery_date=Timestamp.from_mjd([60050, 60250, 60300]),
    )

    # Create test results
    results = WindowResult.from_kwargs(
        orbit_id=["test1", "test1", "test2", "test2", "test3", "test3"],
        condition_id=[
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
        ],
        status=["complete", "complete", "complete", "complete", "complete", "complete"],
        observation_start=Timestamp.from_mjd(
            [60000, 60010, 60000, 60000, 60000, 60000]
        ),
        observation_end=Timestamp.from_mjd([60050, 60060, 60150, 60250, 60250, 60300]),
        window=[
            "60000_60050",
            "60000_60060",
            "60000_60150",
            "60000_60250",
            "60000_60250",
            "60000_60300",
        ],
        observation_count=[10, 15, 5, 20, 10, 10],
        observations_rejected=[0, 0, 0, 0, 0, 0],
        observation_nights=[1, 2, 1, 2, 1, 2],
        impact_probability=[
            0.00001,
            0.3,
            0.00001,
            0.00002,
            0.5,
            0.5,
        ],
    )

    # Compute warning times
    warning_times = compute_warning_time(
        impactor_orbits, results, discovery_dates, threshold=1e-4
    )

    assert len(warning_times) == 3

    # Check object IDs are present
    assert pc.any(pc.equal(warning_times.orbit_id, "test1")).as_py()
    assert pc.any(pc.equal(warning_times.orbit_id, "test2")).as_py()
    assert pc.any(pc.equal(warning_times.orbit_id, "test3")).as_py()

    # Check warning time
    # obj1 doesn't meet threshold until 60060
    warning_time_obj1 = warning_times.select("orbit_id", "test1")
    assert warning_time_obj1.warning_time[0].as_py() == 40.0  # 60100 - 60060

    # obj2 doesn't meet threshold at all
    warning_time_obj2 = warning_times.select("orbit_id", "test2")
    assert pc.all(pc.is_null(warning_time_obj2.warning_time)).as_py()

    # obj3 meets threshold at 60250 but isn't technically discovered until 60300
    warning_time_obj3 = warning_times.select("orbit_id", "test3")
    assert (
        warning_time_obj3.warning_time[0].as_py() == 50.0
    )  # 60350 - 60300 (discovery date)

    # Make sure warning time still works if inputs are not sorted
    scrambled_results = results.take([1, 0, 4, 2, 5, 3])
    scrambled_impactor_orbits = impactor_orbits.take([1, 2, 0])
    scrambled_discovery_dates = discovery_dates.take([2, 1, 0])
    warning_times = compute_warning_time(
        scrambled_impactor_orbits,
        scrambled_results,
        scrambled_discovery_dates,
        threshold=0.25,
    )
    assert len(warning_times) == 3
    assert warning_times.orbit_id.to_pylist() == ["test1", "test2", "test3"]
    assert warning_times.warning_time.to_pylist() == [40.0, None, 50.0]


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
    discovery_dates = DiscoveryDates.from_kwargs(
        orbit_id=["test1", "test2", "test3"],
        discovery_date=Timestamp.from_mjd([60050, 60250, 60350]),
    )
    empty_results = WindowResult.empty()

    empty_warning_times = compute_warning_time(
        impactor_orbits, empty_results, discovery_dates
    )
    assert len(empty_warning_times) == 3
    assert pc.all(pc.is_null(empty_warning_times.column("warning_time"))).as_py()

    # Test all probabilities below threshold
    low_prob_results = WindowResult.from_kwargs(
        orbit_id=["obj1"],
        condition_id=["Default - Earth"],
        status=["complete"],
        observation_start=Timestamp.from_mjd([60000]),
        observation_end=Timestamp.from_mjd([60050]),
        window=["60000_60050"],
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

    low_prob_warning_times = compute_warning_time(
        low_prob_orbits, low_prob_results, discovery_dates
    )
    assert len(low_prob_warning_times) == 1


def test_compute_ip_threshold_date():
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
        impact_time=Timestamp.from_mjd([60500, 60300, 60350]),
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
        orbit_id=["test1", "test1", "test1", "test2", "test1", "test1"],
        condition_id=[
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
        ],
        status=["complete", "complete", "complete", "complete", "complete", "complete"],
        observation_start=Timestamp.from_mjd(
            [60000, 60000, 60000, 60000, 60000, 60000], scale="utc"
        ),
        observation_end=Timestamp.from_mjd(
            [60050, 60060, 60070, 60080, 60090, 60100], scale="utc"
        ),
        window=[
            "60000_60050",
            "60000_60060",
            "60000_60070",
            "60000_60080",
            "60000_60090",
            "60000_60100",
        ],
        observation_count=[10, 15, 20, 25, 30, 35],
        observations_rejected=[0, 0, 0, 0, 0, 0],
        observation_nights=[1, 1, 1, 1, 1, 1],
        impact_probability=[0.0001, 0.01, 0.3, 0.99, 0.3, 0.3],
    )

    # use take to randomly sort the results so we know our sorting and drop duplicates is working

    ip_threshold_date_1_percent = compute_ip_threshold_date(
        impactor_orbits, results, 0.01
    )
    assert ip_threshold_date_1_percent.sort_by("orbit_id").date.mjd().to_pylist() == [
        60060.0,
        60080.0,
        None,
    ]

    ip_threshold_date_30_percent = compute_ip_threshold_date(
        impactor_orbits, results, 0.3
    )
    assert ip_threshold_date_30_percent.sort_by("orbit_id").date.mjd().to_pylist() == [
        60070.0,
        60080.0,
        None,
    ]

    ip_threshold_date_90_percent = compute_ip_threshold_date(
        impactor_orbits, results, 0.9
    )
    assert ip_threshold_date_90_percent.sort_by("orbit_id").date.mjd().to_pylist() == [
        None,
        60080.0,
        None,
    ]


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
        condition_id=[
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
            "Default - Earth",
        ],
        status=["complete", "complete", "complete", "complete"],
        observation_start=Timestamp.from_mjd([60000, 60010, 60000, 60000]),
        observation_end=Timestamp.from_mjd([60050, 60075, 60150, 60250]),
        window=["60000_60050", "60000_60075", "60000_60150", "60000_60250"],
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

    # Now test with scrambled inputs
    scrambled_results = results.take([1, 0, 2, 3])
    scrambled_impactor_orbits = impactor_orbits.take([1, 0, 2])
    realization_times = compute_realization_time(
        scrambled_impactor_orbits, scrambled_results, discovery_dates, threshold=1e-9
    )
    assert len(realization_times) == 3
    assert realization_times.orbit_id.to_pylist() == ["test1", "test2", "test3"]
    assert realization_times.realization_time.to_pylist() == [25.0, None, 0.0]
