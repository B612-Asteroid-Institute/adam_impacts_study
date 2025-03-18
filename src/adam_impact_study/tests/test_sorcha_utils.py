import os
import pathlib
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
from adam_core.coordinates import CartesianCoordinates, KeplerianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.sorcha_utils import (
    PhotometricProperties,
    run_sorcha,
    write_config_file_timeframe,
    write_phys_params_file,
    write_sorcha_orbits_file,
)
from adam_impact_study.types import ImpactorOrbits


def test_write_config_file_timeframe(tmpdir):
    assist_min_dt = 1e-9
    assist_initial_dt = 1e-6
    assist_epsilon = 1e-9
    assist_adaptive_mode = 1
    impact_date = Timestamp.from_mjd([59580.0], scale="utc")
    config_file = tmpdir.join("config.txt")
    written_file = write_config_file_timeframe(
        config_file,
        impact_date,
        assist_epsilon,
        assist_min_dt,
        assist_initial_dt,
        assist_adaptive_mode,
    )

    with open(written_file, "r") as f:
        content = f.read()

    assert os.path.exists(written_file)

    pointing_command = (
        f"SELECT observationId, observationStartMJD as observationStartMJD_TAI, visitTime, visitExposureTime, filter, "
        f"seeingFwhmGeom as seeingFwhmGeom_arcsec, seeingFwhmEff as seeingFwhmEff_arcsec, fiveSigmaDepth as fieldFiveSigmaDepth_mag , "
        f"fieldRA as fieldRA_deg, fieldDec as fieldDec_deg, rotSkyPos as fieldRotSkyPos_deg FROM observations "
        f"WHERE observationStartMJD < {impact_date.rescale('tai').mjd()[0].as_py()} ORDER BY observationId"
    )

    print("pointing_command")
    print(pointing_command)
    print("content")
    print(content)

    assert pointing_command in content

    assert f"ar_min_dt = {assist_min_dt}" in content
    assert f"ar_initial_dt = {assist_initial_dt}" in content
    assert f"ar_epsilon = {assist_epsilon}" in content
    assert f"ar_adaptive_mode = {assist_adaptive_mode}" in content


@pytest.fixture
def mock_orbits():
    cartesian_coords = CartesianCoordinates.from_kwargs(
        x=[0.1],
        y=[0.2],
        z=[-0.3],
        vx=[-0.01],
        vy=[0.02],
        vz=[0.03],
        time=Timestamp.from_mjd([59000], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["Object1"], object_id=["Object1"], coordinates=cartesian_coords
    )
    return orbits


@pytest.fixture
def mock_impactor_orbits(mock_orbits):
    impactor_orbits = ImpactorOrbits.from_kwargs(
        orbit_id=["ObjA", "ObjB", "ObjC"],
        object_id=["ObjA", "ObjB", "ObjC"],
        coordinates=mock_orbits.coordinates.take([0, 0, 0]),
        impact_time=Timestamp.from_mjd([60000, 60000, 60000], scale="tai"),
        dynamical_class=["Apollo", "Apollo", "Apollo"],
        ast_class=["S", "C", "S"],
        diameter=[0.01, 0.1, 1],
        albedo=[0.1, 0.2, 0.3],
        H_r=[15.01, 16.012, 17.03],
        u_r=[1.71, 1.72, 1.73],
        g_r=[0.50, 0.51, 0.52],
        i_r=[-0.11, -0.12, -0.13],
        z_r=[-0.11, -0.12, -0.13],
        y_r=[-0.11, -0.12, -0.13],
        GS=[0.15, 0.16, 0.17],
    )
    return impactor_orbits


def test_write_sorcha_orbits_file(mock_orbits, tmpdir):
    sorcha_orbits_file = tmpdir.join("sorcha_orbits.csv")
    write_sorcha_orbits_file(mock_orbits, sorcha_orbits_file)
    impactor_table = pa.csv.read_csv(
        sorcha_orbits_file, parse_options=pa.csv.ParseOptions(delimiter=" ")
    )

    assert len(impactor_table) == 1
    assert impactor_table["ObjID"][0].as_py() == "Object1"

    keplerian_coords = KeplerianCoordinates.from_kwargs(
        a=impactor_table["a"].to_numpy(zero_copy_only=False),
        e=impactor_table["e"].to_numpy(zero_copy_only=False),
        i=impactor_table["inc"].to_numpy(zero_copy_only=False),
        raan=impactor_table["node"].to_numpy(zero_copy_only=False),
        ap=impactor_table["argPeri"].to_numpy(zero_copy_only=False),
        M=impactor_table["ma"].to_numpy(zero_copy_only=False),
        time=Timestamp.from_mjd(
            impactor_table["epochMJD_TDB"].to_numpy(zero_copy_only=False), scale="tdb"
        ),
        origin=Origin.from_kwargs(
            code=np.full(len(impactor_table), "SUN", dtype="object")
        ),
        frame="ecliptic",
    )

    cartesian_coords = keplerian_coords.to_cartesian()

    assert (cartesian_coords.x[0].as_py() - 0.1) < 1e-10
    assert (cartesian_coords.y[0].as_py() - 0.2) < 1e-10
    assert (cartesian_coords.z[0].as_py() + 0.3) < 1e-10
    assert (cartesian_coords.vx[0].as_py() + 0.01) < 1e-10
    assert (cartesian_coords.vy[0].as_py() - 0.02) < 1e-10
    assert (cartesian_coords.vz[0].as_py() - 0.03) < 1e-10


@pytest.fixture
def mock_photometric_properties() -> PhotometricProperties:
    data = {
        "orbit_id": ["ObjA", "ObjB", "ObjC"],
        "H_mf": [15.01, 16.012, 17.03],
        "u_mf": [1.71, 1.72, 1.73],
        "g_mf": [0.50, 0.51, 0.52],
        "i_mf": [-0.11, -0.12, -0.13],
        "z_mf": [-0.11, -0.12, -0.13],
        "y_mf": [-0.11, -0.12, -0.13],
        "GS": [0.15, 0.16, 0.17],
    }
    return PhotometricProperties.from_kwargs(**data)


def test_write_phys_params_file(tmpdir, mock_photometric_properties):
    sorcha_physical_params_file = os.path.join(tmpdir, "physical_params.txt")
    write_phys_params_file(
        mock_photometric_properties, sorcha_physical_params_file, filter_band="r"
    )
    assert os.path.exists(sorcha_physical_params_file)

    read_table = pa.csv.read_csv(
        sorcha_physical_params_file, parse_options=pa.csv.ParseOptions(delimiter=" ")
    )
    expected_table = pa.table(
        {
            "ObjID": ["ObjA", "ObjB", "ObjC"],
            "H_r": [15.01, 16.012, 17.03],
            "u-r": [1.71, 1.72, 1.73],
            "g-r": [0.50, 0.51, 0.52],
            "i-r": [-0.11, -0.12, -0.13],
            "z-r": [-0.11, -0.12, -0.13],
            "y-r": [-0.11, -0.12, -0.13],
            "GS": [0.15, 0.16, 0.17],
        }
    )
    assert expected_table.equals(read_table)


@patch("adam_impact_study.sorcha_utils.write_sorcha_orbits_file")
@patch("adam_impact_study.sorcha_utils.write_phys_params_file")
@patch("adam_impact_study.sorcha_utils.write_config_file_timeframe")
@patch("adam_impact_study.sorcha_utils.tempfile.mkdtemp")
@patch("subprocess.run")
def test_run_sorcha(
    mock_subprocess,
    mock_mkdtemp,
    mock_config,
    mock_params,
    mock_orbits_write,
    mock_impactor_orbits,
    tmpdir,
):
    """Test that run_sorcha calls all the necessary functions with correct arguments"""
    single_impactor = mock_impactor_orbits.take([0])
    pointing_file = str(tmpdir.join("pointing.db"))
    working_dir = str(tmpdir.mkdir("working"))

    hardcoded_tmp_dir = tmpdir.mkdir("sorcha_tmp")
    mock_mkdtemp.return_value = hardcoded_tmp_dir
    seed = 612

    assist_min_dt = 1e-9
    assist_initial_dt = 1e-6
    assist_epsilon = 1e-9
    assist_adaptive_mode = 1

    run_sorcha(
        single_impactor,
        single_impactor.impact_time.add_days(-1),
        pointing_file,
        working_dir,
        assist_epsilon,
        assist_min_dt,
        assist_initial_dt,
        assist_adaptive_mode,
        seed,
    )

    # Verify the file writing functions were called correctly
    mock_orbits_write.assert_called_once_with(
        single_impactor.orbits(), pathlib.Path(f"{hardcoded_tmp_dir}/orbits.csv")
    )
    mock_params.assert_called_once_with(
        single_impactor.photometric_properties(),
        pathlib.Path(f"{hardcoded_tmp_dir}/params.csv"),
        filter_band="r",
    )
    mock_config.assert_called_once_with(
        pathlib.Path(f"{hardcoded_tmp_dir}/config.ini"),
        single_impactor.impact_time.add_days(-1),
        assist_epsilon,
        assist_min_dt,
        assist_initial_dt,
        assist_adaptive_mode,
    )

    # Check the sorcha command is correct
    expected_command = (
        f"SORCHA_SEED={seed} "
        f"sorcha run -c {hardcoded_tmp_dir}/config.ini -p {hardcoded_tmp_dir}/params.csv "
        f"--orbits {hardcoded_tmp_dir}/orbits.csv --pointing-db {pointing_file} "
        f"-o {hardcoded_tmp_dir} --stem observations -f"
    )
    mock_subprocess.assert_called_once_with(expected_command, shell=True)
