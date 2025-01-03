from unittest.mock import patch

import pandas as pd
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.dynamics.impacts import ImpactProbabilities
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_assist import ASSISTPropagator
from adam_core.time import Timestamp

from adam_impact_study.conversions import (
    Observations,
    Photometry,
    impactor_file_to_adam_orbit,
)
from adam_impact_study.impacts_study import run_impact_study_all


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


@patch("adam_impact_study.impacts_study.calculate_impact_probabilities")
@patch("adam_impact_study.impacts_study.calculate_impacts")
@patch("adam_impact_study.impacts_study.ASSISTPropagator")
@patch("adam_impact_study.impacts_study.run_fo_od")
@patch("adam_impact_study.impacts_study.run_sorcha")
def test_run_impact_study_fo(
    mock_run_sorcha,
    mock_run_fo_od,
    mock_propagator,
    mock_calculate_impacts,
    mock_calculate_impact_probabilities,
    tmpdir,
):
    impactors_file = tmpdir.join("impactors.csv")
    pointing_file = tmpdir.join("pointing_file.txt")

    RUN_NAME = "Impact_Study_Test"
    FO_DIR = tmpdir.mkdir("FO_DIR")
    RUN_DIR = tmpdir.mkdir("RUN_DIR")
    RESULT_DIR = tmpdir.mkdir("RESULT_DIR")

    csv_data = """ObjID,q_au,e,i_deg,argperi_deg,node_deg,tp_mjd,epoch_mjd,H_mag,a_au,M_deg
I00000,0.9346171379884184,0.3895326794095313,7.566861357949266,38.66627303305196,179.34855525994243,66264.30470146648,66202.91220580554,24.931599051023323,1.5309863549852576,328.05797964887734
I00001,0.9125315468414172,0.3841166640887326,2.1597232256169803,42.129078921761604,100.19335181650827,61804.80697714385,61741.401768986136,24.999606842888358,1.481662993026473,325.34987099452826"""
    impactors_file = tmpdir.join("Impactors.csv")
    impactors_file.write(csv_data)

    orbits = impactor_file_to_adam_orbit(impactors_file)

    config_data = """
{
  "C_albedo_min": 0.03,
  "C_albedo_max": 0.09,
  "S_albedo_min": 0.10,
  "S_albedo_max": 0.22,
  "percent_C": 0.5,
  "percent_S": 0.5,
  "min_diam": 0.001,
  "max_diam": 100,
  "n_asteroids": 1000,
  "u_r_C": 1.786,
  "g_r_C": 0.474,
  "i_r_C": -0.119,
  "z_r_C": -0.126,
  "y_r_C": -0.131,
  "u_r_S": 2.182,
  "g_r_S": 0.65,
  "i_r_S": -0.2,
  "z_r_S": -0.146,
  "y_r_S": -0.151
}
"""
    run_config_file = tmpdir.join("run_config.json")
    run_config_file.write(config_data)

    # Mock returns
    mock_calculate_impact_probabilities.return_value = ImpactProbabilities.from_kwargs(
        orbit_id=["1", "2", "3"],
        impacts=[1, 2, 0],
        variants=[3, 3, 3],
        cumulative_probability=[1 / 3, 2 / 3, 0.0],
    )

    mock_calculate_impacts.return_value = [None, None]

    mock_run_sorcha.return_value = Observations.from_kwargs(
        obs_id=["obs1", "obs2", "obs3", "obs4", "obs5"],
        object_id=["Test_1001", "Test_1001", "Test_1001", "Test_1002", "Test_1002"],
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
    )

    cartesian_coords = CartesianCoordinates.from_kwargs(
        x=[2.7003],
        y=[-0.45319],
        z=[0.065459],
        vx=[0.00013123],
        vy=[0.0015833],
        vz=[-0.0000083965],
        time=Timestamp.from_mjd([60200.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["Object1"], object_id=["Object1"], coordinates=cartesian_coords
    )

    mock_run_fo_od.return_value = orbits

    # Call the function with the mocked inputs
    try:
        run_impact_study_all(
            orbits,
            str(run_config_file),
            str(pointing_file),
            str(RUN_NAME),
            str(FO_DIR),
            str(RUN_DIR),
            str(RESULT_DIR),
        )
    except Exception as e:
        pytest.fail(f"run_impact_study_fo raised an exception: {e}")

    mock_run_sorcha.assert_called()
    mock_run_fo_od.assert_called()
    mock_calculate_impact_probabilities.assert_called()
