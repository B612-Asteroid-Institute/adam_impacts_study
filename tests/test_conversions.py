import json
import os

import pytest
from adam_core.coordinates import Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.conversions import (
    Observations,
    Photometry,
    fo_to_adam_orbit_cov,
    impactor_file_to_adam_orbit,
    od_observations_to_ades_file,
    read_fo_output,
    sorcha_output_to_od_observations,
)


@pytest.fixture
def mock_impactor_file(tmpdir):
    file_path = tmpdir.join("impactor_file.csv")
    impactor_data = """ObjID,q_au,e,i_deg,argperi_deg,node_deg,tp_mjd,epoch_mjd,H_mag,a_au,M_deg
Test_1001,0.1111111111111111,0.1111111111111111,10.123456789123456,110.12345678912345,210.12345678912345,65001.123456789123,59001.123456789123,24.123456789123456,1.1234567891234567,10.123456789123456
Test_1002,0.2222222222222222,0.2222222222222222,20.123456789123456,120.12345678912345,220.12345678912345,65002.123456789123,59002.123456789123,25.123456789123456,2.1234567891234567,20.123456789123456
"""
    file_path.write(impactor_data)
    return str(file_path)


def test_impactor_file_to_adam_orbit(mock_impactor_file):
    orbits = impactor_file_to_adam_orbit(mock_impactor_file)

    assert isinstance(orbits, Orbits)
    assert len(orbits) == 2
    assert orbits.orbit_id[0].as_py() == "Test_1001"
    assert orbits.orbit_id[1].as_py() == "Test_1002"

    # Conver back to keplerian and check the values
    coords_kep = orbits.coordinates.to_keplerian()
    assert (coords_kep.a[0].as_py() - 1.1234567891234567) < 1e-13
    assert (coords_kep.a[1].as_py() - 2.1234567891234567) < 1e-13
    assert (coords_kep.e[0].as_py() - 0.1111111111111111) < 1e-13
    assert (coords_kep.e[1].as_py() - 0.2222222222222222) < 1e-13
    assert (coords_kep.i[0].as_py() - 10.123456789123456) < 1e-13
    assert (coords_kep.i[1].as_py() - 20.123456789123456) < 1e-13
    assert (coords_kep.raan[0].as_py() - 210.12345678912345) < 1e-13
    assert (coords_kep.raan[1].as_py() - 220.12345678912345) < 1e-13
    assert (coords_kep.ap[0].as_py() - 110.12345678912345) < 1e-13
    assert (coords_kep.ap[1].as_py() - 120.12345678912345) < 1e-13
    assert (coords_kep.M[0].as_py() - 10.123456789123456) < 1e-13
    assert (coords_kep.M[1].as_py() - 20.123456789123456) < 1e-13
    assert coords_kep.time.mjd()[0].as_py() == 59001.123456789123
    assert coords_kep.time.mjd()[1].as_py() == 59002.123456789123


@pytest.fixture
def mock_sorcha_output_file(tmpdir):
    file_path = tmpdir.join("sorcha_output.csv")
    sorcha_data = """ObjID,fieldMJD_TAI,fieldRA_deg,fieldDec_deg,RA_deg,Dec_deg,astrometricSigma_deg,optFilter,trailedSourceMag,trailedSourceMagSigma,fiveSigmaDepth_mag,phase_deg
Test_1001,60001.12345678912,340.1234567,-7.1234567,341.1234567,-8.1234567,1.12e-05,i,21.123,0.123,22.123,18.12345678912345
Test_1001,60002.12345678912,341.1234567,-6.1234567,342.1234567,-7.1234567,2.12e-05,r,21.123,0.123,23.123,19.12345678912345
Test_1001,60003.12345678912,342.1234567,-5.1234567,343.1234567,-6.1234567,3.12e-05,z,21.123,0.123,24.123,20.12345678912345
Test_1002,60005.12345678912,344.1234567,-4.1234567,345.1234567,-5.1234567,8.12e-06,r,22.123,0.123,24.123,20.12345678912345
Test_1002,60006.12345678912,345.1234567,-3.1234567,346.1234567,-4.1234567,9.12e-06,i,23.123,0.123,25.123,21.12345678912345
"""
    file_path.write(sorcha_data)
    return str(file_path)


def test_sorcha_output_to_od_observations(mock_sorcha_output_file):
    observations = sorcha_output_to_od_observations(mock_sorcha_output_file)
    assert observations is not None
    assert len(observations) == 5
    assert observations.object_id.to_pylist() == [
        "Test_1001",
        "Test_1001",
        "Test_1001",
        "Test_1002",
        "Test_1002",
    ]
    assert observations.coordinates.lon.to_pylist() == [
        341.1234567,
        342.1234567,
        343.1234567,
        345.1234567,
        346.1234567,
    ]
    assert observations.coordinates.lat.to_pylist() == [
        -8.1234567,
        -7.1234567,
        -6.1234567,
        -5.1234567,
        -4.1234567,
    ]
    # Rescale times to TAI
    times = Timestamp.from_mjd(observations.coordinates.time.mjd(), scale="utc")
    times_tai = times.rescale("tai")
    assert times_tai.mjd().to_pylist() == [
        60001.12345678912,
        60002.12345678912,
        60003.12345678912,
        60005.12345678912,
        60006.12345678912,
    ]


@pytest.fixture
def mock_observations():
    return Observations.from_kwargs(
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


def test_od_observations_to_ades_file(mock_observations, tmpdir):
    ades_file_path = tmpdir.join("ades_output.xml")
    output_file = od_observations_to_ades_file(mock_observations, str(ades_file_path))
    assert os.path.exists(output_file)

    # Read and check the contents of the ADES file
    with open(output_file, "r") as file:
        contents = file.read()
        assert "Vera C. Rubin Observatory" in contents
        print(contents)
        assert "trkSub|obsTime|ra|dec|mag|rmsMag|band|stn|mode|astCat" in contents
        assert (
            "Test_1001|2023-02-26T00:00:00.000Z|180.00000000|0.00000000|21.00|0.10|i|X05|NA|NA"
            in contents
        )
        assert (
            "Test_1001|2023-02-27T00:00:00.000Z|181.00000000|1.00000000|22.00|0.20|r|X05|NA|NA"
            in contents
        )
        assert (
            "Test_1001|2023-02-28T00:00:00.000Z|182.00000000|2.00000000|23.00|0.30|z|X05|NA|NA"
            in contents
        )
        assert (
            "Test_1002|2023-03-02T00:00:00.000Z|183.00000000|3.00000000|24.00|0.40|r|X05|NA|NA"
            in contents
        )
        assert (
            "Test_1002|2023-03-03T00:00:00.000Z|184.00000000|4.00000000|25.00|0.50|i|X05|NA|NA"
            in contents
        )


@pytest.fixture
def mock_fo_output_files(tmpdir):
    total_file = tmpdir.join("total.json")
    covar_file = tmpdir.join("covar.json")

    total_data = {"num": 1,
  "ids":
  [
    "Test_1001" 
  ],
  "objects":
  {
    "Test_1001":
    {
      "object": "Test_1001",
      "packed": "Test_1001",
      "created": 2460558.23098,
      "created iso": "2024-09-04T17:32:36Z",
      "Find_Orb_version": 2460533.50000,
      "Find_Orb_version_iso": "2024-08-11T00:00:00Z",
      "elements":
      {
        "central body": "Sun",
        "frame": "J2000 ecliptic",
        "reference": "Find_Orb",
        "epoch_iso": "2024-09-29T00:00:00Z",
        "epoch":  2460582.50000000,
        "P": 1394.9253892736097, "P sigma": 0.00253,
        "M":  77.3353733269255, "M sigma": 0.00015,
        "n":   0.2580783192909, "n sigma": 4.68e-7,
        "a":   2.4432525915365, "a sigma": 2.95e-6,
        "e":   0.9842422700762, "e sigma": 3.12e-7,
        "q":   0.0385001144730, "q sigma": 7.83e-7,
        "Q":   4.8480050685999, "Q sigma": 5.57e-6,
        "i": 179.9930186752411, "i sigma": 0.00006,
        "arg_per":  189.7896453405323, "arg_per sigma":  0.031,
        "asc_node": 321.7381294352738, "asc_node sigma": 0.031,
        "Tp": 2460282.84145627, "Tp sigma": 0.000342,
        "Tp_iso": "2023-12-04T08:11:41.821Z",
        "H":  16.06,
        "G":   0.15,
        "rms_residual": 0.033146,
        "weighted_rms_residual": 0.9631,
        "n_resids": 102,
        "U": 2.9896,
        "p_NEO": 100.0000,
        "MOIDs":
        {
          "Mercury" : 0.038916,
          "Venus" : 0.022080,
          "Earth" : 0.000009,
          "Mars" : 0.041731,
          "Jupiter" : 0.237747,
          "Saturn" : 5.066341,
          "Uranus" : 14.993002,
          "Neptune" : 25.227115 
        }
      },
      "observations":
      {}
    }
    }
    }

    covar_data = {
        "covar": [
            [
                4.16121327342e-12,
                -2.09134780573e-12,
                -2.30283659349e-16,
                -2.71788684422e-14,
                -1.04967286688e-14,
                1.23426143706e-18
            ],
            [
                -2.09134780573e-12,
                1.06124575034e-12,
                3.56294827847e-17,
                1.35942552795e-14,
                5.18898046367e-15,
                6.94693054473e-19
            ],
            [
                -2.30283659349e-16,
                3.56294827847e-17,
                5.18082512922e-15,
                4.41513579676e-18,
                -1.11430144937e-18,
                -9.37068796782e-17
            ],
            [
                -2.71788684422e-14,
                1.35942552795e-14,
                4.41513579676e-18,
                2.08874755856e-16,
                3.6370326027e-17,
                -8.68973770791e-20
            ],
            [
                -1.04967286688e-14,
                5.18898046367e-15,
                -1.11430144937e-18,
                3.6370326027e-17,
                6.53394971075e-17,
                2.33681396804e-20
            ],
            [
                1.23426143706e-18,
                6.94693054473e-19,
                -9.37068796782e-17,
                -8.68973770791e-20,
                2.33681396804e-20,
                2.54650689185e-18
            ]
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
    }

    total_file.write(json.dumps(total_data))
    covar_file.write(json.dumps(covar_data))

    return str(total_file), str(covar_file)


def test_read_fo_output(mock_fo_output_files, tmpdir):
    elements, covariances = read_fo_output(tmpdir)
    assert elements is not None
    assert covariances is not None


def test_fo_to_adam_orbit_cov(mock_fo_output_files, tmpdir):
    orbits = fo_to_adam_orbit_cov(tmpdir)
    assert isinstance(orbits, Orbits)
    assert orbits is not None
    assert orbits.orbit_id[0].as_py() == "Test_1001"
    assert len(orbits) == 1
    assert orbits.coordinates.x[0].as_py() == 2.40431779633740117
    assert orbits.coordinates.y[0].as_py() == -2.0586498601048886
    assert orbits.coordinates.z[0].as_py() == 1.56463294002342372e-05
    assert orbits.coordinates.vx[0].as_py() == 0.00508148904172802708
    assert orbits.coordinates.vy[0].as_py() == -0.00632766941087369462
    assert orbits.coordinates.vz[0].as_py() == 2.1947140448603267e-07
    assert orbits.coordinates.time.mjd()[0].as_py() - 60490.342573 < 1e-6
    assert orbits.coordinates.covariance.values[0][0].as_py() - 4.16121327342e-12 < 1e-13