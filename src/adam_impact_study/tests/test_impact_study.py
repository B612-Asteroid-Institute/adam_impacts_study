import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pytest
from adam_assist import ASSISTPropagator
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from adam_core.dynamics.impacts import EarthImpacts, ImpactProbabilities
from adam_core.observations.ades import ADESObservations
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.conversions import Observations, Photometry
from adam_impact_study.impacts_study import run_impact_study_for_orbit
from adam_impact_study.types import ImpactorOrbits, WindowResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def impactors_file_mock(tmpdir):
    csv_data = """ObjID,q_au,e,i_deg,argperi_deg,node_deg,tp_mjd,epoch_mjd,H_mag,a_au,M_deg
I00000,0.9346171379884184,0.3895326794095313,7.566861357949266,38.66627303305196,179.34855525994243,66264.30470146648,66202.91220580554,24.931599051023323,1.5309863549852576,328.05797964887734
I00001,0.9125315468414172,0.3841166640887326,2.1597232256169803,42.129078921761604,100.19335181650827,61804.80697714385,61741.401768986136,24.999606842888358,1.481662993026473,325.34987099452826
I00002,0.8676492513962372,0.09298572306135865,15.634938862271255,225.73585788385364,251.68582512123703,63241.74420855354,62989.86858265847,24.96677474326819,0.9565993319584019,94.66465437008878
I00003,0.9803020879510772,0.6454041054450625,34.23444883685873,338.8162670022895,347.50260746689463,64564.318086652595,64550.87758333795,24.998410549502168,2.764561302046319,357.1180909593566
I00004,0.259627599479878,0.6051897273788432,4.754471135236431,167.17731145750275,290.2897847502658,60944.22220318081,60838.79582554242,24.955862876736067,0.6576009224790503,165.14581490588978
I00005,0.98365770653899,0.4325266511756645,4.450125108830237,5.12543912622952,128.48187477300706,63629.79965376174,63595.52372706169,24.959473208429984,1.7333989491786452,345.1971449722603
I00006,0.4893246473666471,0.6343876719347437,0.9086995763471053,107.09962415142384,133.95972365327026,64777.41796440458,64696.53924952028,24.979805494380756,1.3383702074709853,308.5158051793827
I00007,0.7157257204929293,0.6218650982519122,10.955610348964763,76.72885508778904,302.71386343673385,60564.26303454751,60486.532906815875,24.949238239710944,1.892778786576393,330.5799324625874
I00008,0.038421632426207344,0.9842751278600509,180.00198320977762,148.53256327065648,280.487516820131,64467.32634705419,63073.85314633604,24.91194226482537,2.4433669211590607,0.4000193726312773
I00009,0.46573276386416124,0.38332295394222854,12.525649549276613,197.2295904835092,293.5042450177829,64302.62567454859,64129.44801568473,24.961840690064147,0.7552296081740822,99.93768364257812
"""
    impactors_file = tmpdir.join("10_impactors.csv")
    impactors_file.write(csv_data)
    return str(impactors_file)


@pytest.fixture
def impactor_orbits():
    cartesian_coords = CartesianCoordinates.from_kwargs(
        x=[2.7003],
        y=[-0.45319],
        z=[0.065459],
        vx=[0.00013123],
        vy=[0.0015833],
        vz=[-0.0000083965],
        time=Timestamp.from_mjd([60200.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        covariance=CoordinateCovariances.from_matrix(
            np.array(
                [
                    [
                        [
                            4.16121327e-12,
                            -2.09134781e-12,
                            -2.30283659e-16,
                            -2.71788684e-14,
                            -1.04967287e-14,
                            1.23426144e-18,
                        ],
                        [
                            -2.09134781e-12,
                            1.06124575e-12,
                            3.56294828e-17,
                            1.35942553e-14,
                            5.18898046e-15,
                            6.94693054e-19,
                        ],
                        [
                            -2.30283659e-16,
                            3.56294828e-17,
                            5.18082513e-15,
                            4.41513580e-18,
                            -1.11430145e-18,
                            -9.37068797e-17,
                        ],
                        [
                            -2.71788684e-14,
                            1.35942553e-14,
                            4.41513580e-18,
                            2.08874756e-16,
                            3.63703260e-17,
                            -8.68973771e-20,
                        ],
                        [
                            -1.04967287e-14,
                            5.18898046e-15,
                            -1.11430145e-18,
                            3.63703260e-17,
                            6.53394971e-17,
                            2.33681397e-20,
                        ],
                        [
                            1.23426144e-18,
                            6.94693054e-19,
                            -9.37068797e-17,
                            -8.68973771e-20,
                            2.33681397e-20,
                            2.54650689e-18,
                        ],
                    ]
                ]
            )
        ),
        frame="ecliptic",
    )
    impactor_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["Object1"],
        object_id=["Object1"],
        coordinates=cartesian_coords,
        impact_time=Timestamp.from_mjd([60300.0], scale="tdb"),
        dynamical_class=["NEO"],
        ast_class=["Aten"],
        diameter=[0.1],
        albedo=[0.15],
        H_r=[21.0],
        u_r=[1.786],
        g_r=[0.474],
        i_r=[-0.119],
        z_r=[-0.126],
        y_r=[-0.131],
        GS=[0.15],
    )
    return impactor_orbits


@pytest.fixture
def pointing_file(tmpdir):
    pointing_file = tmpdir.join("pointing_file.txt")
    # Write th
    return str(pointing_file)


@pytest.fixture
def sorcha_observations():
    return Observations.from_kwargs(
        obs_id=["obs1", "obs2", "obs3", "obs4", "obs5"],
        orbit_id=["Test_1001", "Test_1001", "Test_1001", "Test_1002", "Test_1002"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=[180.0, 181.0, 182.0, 183.0, 184.0],
            lat=[0.0, 1.0, 2.0, 3.0, 4.0],
            time=Timestamp.from_mjd([60001, 60002, 60003, 60005, 60006], scale="utc"),
            origin=Origin.from_kwargs(code=["X05", "X05", "X05", "X05", "X05"]),
            frame="equatorial",
        ),
        observers=Observers.from_code(
            "X05", Timestamp.from_mjd([60001, 60002, 60003, 60005, 60006], scale="utc")
        ),
        photometry=Photometry.from_kwargs(
            mag=[21.0, 22.0, 23.0, 24.0, 25.0],
            mag_sigma=[0.1, 0.2, 0.3, 0.4, 0.5],
            filter=["i", "r", "z", "r", "i"],
        ),
        observing_night=[60000, 60001, 60002, 60004, 60005],
    )


@patch("adam_impact_study.impacts_study.calculate_impact_probabilities")
@patch("adam_impact_study.impacts_study.ASSISTPropagator")
@patch("adam_impact_study.impacts_study.run_fo_od")
@patch("adam_impact_study.impacts_study.run_sorcha")
def test_run_impact_study_for_orbit(
    mock_run_sorcha,
    mock_run_fo_od,
    mock_propagator,
    mock_calculate_impact_probabilities,
    pointing_file,
    impactor_orbits,
    tmpdir,
    sorcha_observations,
):
    """
    Ensure that all the correct functions are getting called.
    """
    RUN_DIR = tmpdir.mkdir("RUN_DIR")

    # Mock returns
    mock_run_sorcha.return_value = sorcha_observations

    cartesian_coords = CartesianCoordinates.from_kwargs(
        x=[2.7003],
        y=[-0.45319],
        z=[0.065459],
        vx=[0.00013123],
        vy=[0.0015833],
        vz=[-0.0000083965],
        time=Timestamp.from_mjd([60200.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        covariance=CoordinateCovariances.from_matrix(
            np.array(
                [
                    [
                        [
                            4.16121327e-12,
                            -2.09134781e-12,
                            -2.30283659e-16,
                            -2.71788684e-14,
                            -1.04967287e-14,
                            1.23426144e-18,
                        ],
                        [
                            -2.09134781e-12,
                            1.06124575e-12,
                            3.56294828e-17,
                            1.35942553e-14,
                            5.18898046e-15,
                            6.94693054e-19,
                        ],
                        [
                            -2.30283659e-16,
                            3.56294828e-17,
                            5.18082513e-15,
                            4.41513580e-18,
                            -1.11430145e-18,
                            -9.37068797e-17,
                        ],
                        [
                            -2.71788684e-14,
                            1.35942553e-14,
                            4.41513580e-18,
                            2.08874756e-16,
                            3.63703260e-17,
                            -8.68973771e-20,
                        ],
                        [
                            -1.04967287e-14,
                            5.18898046e-15,
                            -1.11430145e-18,
                            3.63703260e-17,
                            6.53394971e-17,
                            2.33681397e-20,
                        ],
                        [
                            1.23426144e-18,
                            6.94693054e-19,
                            -9.37068797e-17,
                            -8.68973771e-20,
                            2.33681397e-20,
                            2.54650689e-18,
                        ],
                    ]
                ]
            )
        ),
        frame="ecliptic",
    )
    orbit = Orbits.from_kwargs(
        orbit_id=["Object1"],
        object_id=["Object1"],
        coordinates=cartesian_coords,
    )
    # Configure mock propagator
    mock_propagator.return_value.propagate.return_value = orbit
    mock_run_fo_od.return_value = (
        impactor_orbits.orbits(),
        ADESObservations.empty(),
        None,
    )

    # mock_calculate_impacts.return_value = (
    #     Orbits.from_kwargs(
    #         orbit_id=["Object1"],
    #         object_id=["Object1"],
    #         coordinates=cartesian_coords,
    #     ),
    #     EarthImpacts.nulls(1),
    # )

    mock_propagator.detect_impacts.return_value = (
        Orbits.from_kwargs(
            orbit_id=["Object1"],
            object_id=["Object1"],
            coordinates=cartesian_coords,
        ),
        EarthImpacts.nulls(1),
    )

    mock_calculate_impact_probabilities.return_value = ImpactProbabilities.from_kwargs(
        orbit_id=["Object1"],
        impacts=[1],
        variants=[3],
        cumulative_probability=[1 / 3],
    )

    # Call the function with the mocked inputs
    window_result, timings = run_impact_study_for_orbit(
        impactor_orbits,
        ASSISTPropagator,
        pointing_file,
        str(RUN_DIR),
        monte_carlo_samples=100,
        assist_epsilon=1e-9,
        assist_min_dt=1e-9,
        assist_initial_dt=1e-6,
        assist_adaptive_mode=1,
        max_processes=1,
        seed=12345,
    )

    mock_run_sorcha.assert_called_with(
        impactor_orbits,
        pointing_file,
        f"{RUN_DIR}/Object1/sorcha",
        assist_epsilon=1e-9,
        assist_min_dt=1e-9,
        assist_initial_dt=1e-6,
        assist_adaptive_mode=1,
        seed=12345,
    )

    # Iterate through the calls to mock_run_fo_od
    expected_fo_call_list = [
        (
            sorcha_observations.apply_mask(
                pc.less_equal(sorcha_observations.observing_night, 60002)
            ),
            f"{RUN_DIR}/Object1/windows/60000_60002/fo",
        ),
        (
            sorcha_observations.apply_mask(
                pc.less_equal(sorcha_observations.observing_night, 60004)
            ),
            f"{RUN_DIR}/Object1/windows/60000_60004/fo",
        ),
        (
            sorcha_observations.apply_mask(
                pc.less_equal(sorcha_observations.observing_night, 60005)
            ),
            f"{RUN_DIR}/Object1/windows/60000_60005/fo",
        ),
    ]
    for i, call in enumerate(mock_run_fo_od.call_args_list):
        # Compare the actual arguments with expected arguments
        actual_args = call.args
        expected_args = expected_fo_call_list[i]
        assert (
            actual_args[0] == expected_args[0]
        ), "fo observations not what was expected"
        assert actual_args[1] == expected_args[1], f"Call {i}: Path mismatch"

    expected_calculate_impacts_calls = [
        (impactor_orbits.orbits(), 130, ASSISTPropagator(), 100, 1, 12345),
        (impactor_orbits.orbits(), 130, ASSISTPropagator(), 100, 1, 12345),
        (impactor_orbits.orbits(), 130, ASSISTPropagator(), 100, 1, 12345),
    ]
    for i, call in enumerate(mock_propagator.detect_impacts.call_args_list):
        args, kwargs = call
        assert args[0] == expected_calculate_impacts_calls[i][0]
        assert args[1] == expected_calculate_impacts_calls[i][1]
        assert isinstance(args[2], ASSISTPropagator)
        assert kwargs["num_samples"] == expected_calculate_impacts_calls[i][3]
        assert kwargs["processes"] == expected_calculate_impacts_calls[i][4]
        assert kwargs["seed"] == expected_calculate_impacts_calls[i][5]

    expected = WindowResult.from_kwargs(
        orbit_id=["Object1", "Object1", "Object1"],
        object_id=["Object1", "Object1", "Object1"],
        window=["60000_60002", "60000_60004", "60000_60005"],
        observation_start=Timestamp.from_mjd([60001, 60001, 60001], scale="utc"),
        observation_end=Timestamp.from_mjd([60003, 60005, 60006], scale="utc"),
        observation_count=[3, 4, 5],
        observation_nights=[3, 4, 5],
        observations_rejected=[0, 0, 0],
        impact_probability=[1 / 3, 1 / 3, 1 / 3],
    )
    # Convert both to pandas DataFrames for easier comparison
    # We drop the runtime columns since they are not deterministic
    results_df = window_result.to_dataframe().drop(
        columns=["od_runtime", "ip_runtime", "window_runtime"]
    )
    expected_df = expected.to_dataframe().drop(
        columns=["od_runtime", "ip_runtime", "window_runtime"]
    )

    pd.testing.assert_frame_equal(results_df, expected_df)
