import os
from unittest.mock import patch

import numpy as np
import pytest
from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observations.ades import ADESObservations
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.fo_od import observations_to_ades, run_fo_od
from adam_impact_study.types import Observations, Photometry


@patch("adam_impact_study.fo_od.fo")
def test_run_fo_od(mock_fo, tmpdir):
    """Ensure we are calling fo with the correct arguments"""
    # Create mock fo_result_dir
    fo_result_dir = str(tmpdir.mkdir("FO_DIR"))

    mock_fo.return_value = (
        Orbits.from_kwargs(
            orbit_id=["Test_1001"],
            coordinates=SphericalCoordinates.from_kwargs(
                lon=[180.0],
                lat=[0.0],
                time=Timestamp.from_mjd([60001], scale="utc"),
                origin=Origin.from_kwargs(code=["X05"]),
                frame="equatorial",
            ),
        ),
        ADESObservations.empty(),
        None,
    )

    # Create mock observations
    observations = Observations.from_kwargs(
        obs_id=["obs1", "obs2", "obs3"],
        orbit_id=["Test_1001", "Test_1001", "Test_1001"],
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

    orbits, rejected, errors = run_fo_od(observations, fo_result_dir)
    assert mock_fo.call_count == 1
    assert mock_fo.call_args[0][0] == observations_to_ades(observations)[0]
    # keyword arguments clean_up=True, output_dir=fo_result_dir
    assert mock_fo.call_args[1]["clean_up"] == False
    assert mock_fo.call_args[1]["out_dir"] == fo_result_dir


def test_observations_to_ades():
    # Create test observations
    observations = Observations.from_kwargs(
        obs_id=["obs1", "obs2", "obs3"],
        orbit_id=["Test_1001", "Test_1001", "Test_1001"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=[180.0, 181.0, 182.0],
            lat=[0, 30, 45],
            time=Timestamp.from_mjd([60001, 60002, 60003], scale="utc"),
            origin=Origin.from_kwargs(code=["X05", "X05", "X05"]),
            frame="equatorial",
            covariance=CoordinateCovariances.from_sigmas(
                np.array(
                    [
                        [0, 1 / 3600, 1 / 3600, 0, 0, 0],
                        [0, 1 / 3600, 1 / 3600, 0, 0, 0],
                        [0, 1 / 3600, 1 / 3600, 0, 0, 0],
                    ]
                )
            ),
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

    # Call the function under test
    ades_string, ades_observations = observations_to_ades(observations)

    # Test that the ADES string is created
    assert isinstance(ades_string, str)
    assert len(ades_string) > 0

    # Test that the ADESObservations object is created correctly
    assert isinstance(ades_observations, ADESObservations)
    assert len(ades_observations) == 3

    # Test specific fields
    assert ades_observations.trkSub[0].as_py() == "Test_100"  # 8-char limit
    assert ades_observations.ra[0].as_py() == 180.0
    assert ades_observations.dec[0].as_py() == 0.0
    assert ades_observations.mag[0].as_py() == 21.0
    assert ades_observations.rmsMag[0].as_py() == 0.1
    assert ades_observations.band[0].as_py() == "i"
    assert ades_observations.stn[0].as_py() == "X05"

    # Test that uncertainties were converted to arcseconds
    # For dec=0°, rmsRACosDec should equal rmsDec (both converted from radians to arcseconds)
    assert ades_observations.rmsRACosDec[0].as_py() == 1
    assert ades_observations.rmsDec[0].as_py() == 1

    assert ades_observations.rmsRACosDec[1].as_py() == 1 * np.cos(np.radians(30))
    assert ades_observations.rmsDec[1].as_py() == 1

    assert ades_observations.rmsRACosDec[2].as_py() == 1 * np.cos(np.radians(45))
    assert ades_observations.rmsDec[2].as_py() == 1
