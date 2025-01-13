import os
from unittest.mock import patch

from adam_core.coordinates import Origin, SphericalCoordinates
from adam_core.observations.ades import ADESObservations
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.types import Observations, Photometry


@patch("subprocess.run")
def test_run_fo_od(mock_subprocess_run, tmpdir):
    # Create mock fo_result_dir
    fo_result_dir = str(tmpdir.mkdir("FO_DIR"))

    # Configure mock subprocess to return success
    mock_subprocess_run.return_value.returncode = 0
    mock_subprocess_run.return_value.stdout = ""
    mock_subprocess_run.return_value.stderr = ""

    # Create mock observations
    observations = Observations.from_kwargs(
        obs_id=["obs1", "obs2", "obs3"],
        object_id=["Test_1001", "Test_1001", "Test_1001"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=[180.0, 181.0, 182.0],
            lat=[0.0, 1.0, 2.0],
            time=Timestamp.from_mjd([60001, 60002, 60003], scale="utc"),
            origin=Origin.from_kwargs(code=["X05", "X05", "X05"]),
            frame="equatorial",
        ),
        observers=Observers.from_code(
            "X05", Timestamp.from_mjd([60001, 60002, 60003], scale="utc")
        ),
        photometry=Photometry.from_kwargs(
            mag=[21.0, 22.0, 23.0],
            mag_sigma=[0.1, 0.2, 0.3],
            filter=["i", "r", "z"],
        ),
    )

    # Create mock output files
    covar_file_path = os.path.join(fo_result_dir, "covar.json")
    total_json_file_path = os.path.join(fo_result_dir, "total.json")

    covar_file_text = """{
        "covar": [
            [4.16121327342e-12, -2.09134780573e-12, -2.30283659349e-16, -2.71788684422e-14, -1.04967286688e-14, 1.23426143706e-18],
            [-2.09134780573e-12, 1.06124575034e-12, 3.56294827847e-17, 1.35942552795e-14, 5.18898046367e-15, 6.94693054473e-19],
            [-2.30283659349e-16, 3.56294827847e-17, 5.18082512922e-15, 4.41513579676e-18, -1.11430144937e-18, -9.37068796782e-17],
            [-2.71788684422e-14, 1.35942552795e-14, 4.41513579676e-18, 2.08874755856e-16, 3.6370326027e-17, -8.68973770791e-20],
            [-1.04967286688e-14, 5.18898046367e-15, -1.11430144937e-18, 3.6370326027e-17, 6.53394971075e-17, 2.33681396804e-20],
            [1.23426143706e-18, 6.94693054473e-19, -9.37068796782e-17, -8.68973770791e-20, 2.33681396804e-20, 2.54650689185e-18]
        ],
        "state_vect": [
            2.40431779633740117,
            -2.0586498601048886,
            1.56463294002342372e-05,
            0.00508148904172802708,
            -0.00632766941087369462,
            2.1947140448603267e-07
        ],
        "epoch": 2460490.842573
    }"""

    total_json_text = """{"num": 1, "ids": ["Test_1001"], "objects": {"Test_1001": {
        "object": "Test_1001", "packed": "Test_1001", "created": 2460558.23098,
        "elements": {
            "epoch": 2460582.50000000,
            "frame": "J2000 ecliptic"
        },
        "observations": {}
    }}}"""

    os.makedirs(os.path.dirname(covar_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(total_json_file_path), exist_ok=True)

    with open(covar_file_path, "w") as f:
        f.write(covar_file_text)
    with open(total_json_file_path, "w") as f:
        f.write(total_json_text)

    # Call the function under test
    orbits, rejected, errors = run_fo_od(observations, fo_result_dir)

    assert errors is None
    assert isinstance(orbits, Orbits)
    assert isinstance(rejected, ADESObservations)
    assert orbits.orbit_id[0].as_py() == "Test_1001"
    assert orbits.coordinates.time.mjd()[0].as_py() - 60490.342573 < 1e-6
    assert orbits.coordinates.x[0].as_py() - 2.40431779633740117 < 1e-13
    assert orbits.coordinates.y[0].as_py() - -2.0586498601048886 < 1e-13
    assert orbits.coordinates.z[0].as_py() - 1.56463294002342372e-05 < 1e-13
    assert orbits.coordinates.vx[0].as_py() - 0.00508148904172802708 < 1e-13
    assert orbits.coordinates.vy[0].as_py() - -0.00632766941087369462 < 1e-13
    assert orbits.coordinates.vz[0].as_py() - 2.1947140448603267e-07 < 1e-13
    assert (
        orbits.coordinates.covariance.values[0][0].as_py() - 4.16121327342e-12 < 1e-13
    )

    # Verify that correct find_orb command was run with the expected environment
    mock_subprocess_run.assert_called()
