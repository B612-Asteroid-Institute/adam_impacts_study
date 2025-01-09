import os
from unittest.mock import patch

import numpy as np
import pandas as pd
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


def test_write_config_file_timeframe(tmpdir):
    impact_date = 59580.0
    config_file = tmpdir.join("config.txt")
    written_file = write_config_file_timeframe(impact_date, config_file)

    with open(written_file, "r") as f:
        content = f.read()

    assert os.path.exists(written_file)

    pointing_command = (
        f"SELECT observationId, observationStartMJD as observationStartMJD_TAI, visitTime, visitExposureTime, filter, "
        f"seeingFwhmGeom as seeingFwhmGeom_arcsec, seeingFwhmEff as seeingFwhmEff_arcsec, fiveSigmaDepth as fieldFiveSigmaDepth_mag , "
        f"fieldRA as fieldRA_deg, fieldDec as fieldDec_deg, rotSkyPos as fieldRotSkyPos_deg FROM observations "
        f"WHERE observationStartMJD < {impact_date} ORDER BY observationId"
    )
    assert pointing_command in content


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
        "ObjID": ["ObjA", "ObjB", "ObjC"],
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
        mock_photometric_properties, sorcha_physical_params_file, main_filter="r"
    )
    assert os.path.exists(sorcha_physical_params_file)

    read_table = pa.csv.read_csv(
        sorcha_physical_params_file, parse_options=pa.csv.ParseOptions(delimiter=" ")
    )
    expected_table = pa.Table.from_pandas(mock_physical_params_df)
    assert expected_table.equals(read_table)


@patch("subprocess.run")
def test_run_sorcha(mock_subprocess_run, tmpdir, mock_orbits, mock_physical_params_df):
    config_file = tmpdir.join("config_file.cfg")
    orbits_file = tmpdir.join("orbits.csv")
    physical_params_file = tmpdir.join("physical_params.csv")
    pointing_file = tmpdir.join("pointing_file.txt")
    output_dir = tmpdir.join("output_dir")

    sorcha_output_dir = output_dir.join("sorcha_output")
    os.makedirs(sorcha_output_dir, exist_ok=True)
    sorcha_output_file = sorcha_output_dir.join("output_file.txt")

    # Write dummy file to simulate sorcha output
    with open(sorcha_output_file, "w") as f:
        f.write(
            "ObjID,fieldMJD_TAI,fieldRA_deg,fieldDec_deg,RA_deg,Dec_deg,astrometricSigma_deg,optFilter,trailedSourceMag,trailedSourceMagSigma,fiveSigmaDepth_mag,phase_deg\n"
        )
        f.write(
            "Test_1001,60001.12345678912,340.1234567,-7.1234567,341.1234567,-8.1234567,1.12e-05,i,21.123,0.123,22.123,18.12345678912345\n"
        )
        f.write(
            "Test_1001,60002.12345678912,341.1234567,-6.1234567,342.1234567,-7.1234567,2.12e-05,r,21.123,0.123,23.123,19.12345678912345\n"
        )
        f.write(
            "Test_1001,60003.12345678912,342.1234567,-5.1234567,343.1234567,-6.1234567,3.12e-05,z,21.123,0.123,24.123,20.12345678912345\n"
        )
        f.write(
            "Test_1002,60005.12345678912,344.1234567,-4.1234567,345.1234567,-5.1234567,8.12e-06,r,22.123,0.123,24.123,20.12345678912345\n"
        )
        f.write(
            "Test_1002,60006.12345678912,345.1234567,-3.1234567,346.1234567,-4.1234567,9.12e-06,i,23.123,0.123,25.123,21.12345678912345\n"
        )

    run_sorcha(
        mock_orbits,
        config_file,
        orbits_file,
        physical_params_file,
        "output_file.txt",
        pointing_file,
        "sorcha_output",
        output_dir,
    )

    # Check that subprocess.run was called with the correct command
    expected_command = f"sorcha run -c {config_file} -p {physical_params_file} -ob {orbits_file} -pd {pointing_file} -o {output_dir}/sorcha_output -t sorcha_output -f"
    mock_subprocess_run.assert_called_once_with(expected_command, shell=True)
