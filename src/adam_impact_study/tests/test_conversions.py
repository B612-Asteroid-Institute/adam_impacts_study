import pytest
from adam_core.coordinates import Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.conversions import (
    impactor_file_to_adam_orbit,
    sorcha_output_to_od_observations,
)
from adam_impact_study.types import Observations, Photometry


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

    # Convert back to keplerian and check the values
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
    assert observations.orbit_id.to_pylist() == [
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
    )
