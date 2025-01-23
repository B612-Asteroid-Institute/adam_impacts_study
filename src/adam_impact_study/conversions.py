import json
import logging
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
    Origin,
    SphericalCoordinates,
)
from adam_core.observations import (
    ADES_to_string,
    ADESObservations,
    ObsContext,
    ObservatoryObsContext,
    SubmitterObsContext,
    TelescopeObsContext,
)
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.types import Observations, Photometry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def impactor_file_to_adam_orbit(impactor_file: str) -> Orbits:
    """
    Generate an ADAM Orbit object from an impactor data file.

    Parameters
    ----------
    impactor_file : str
        Path to the impactor data file.

    Returns
    -------
    orbit : `~adam_core.orbits.orbits.Orbits`
        ADAM Orbit object created from the input data.
    """
    impactor_table = pa.csv.read_csv(impactor_file)
    keplerian_coords = KeplerianCoordinates.from_kwargs(
        a=impactor_table["a_au"].to_numpy(zero_copy_only=False),
        e=impactor_table["e"].to_numpy(zero_copy_only=False),
        i=impactor_table["i_deg"].to_numpy(zero_copy_only=False),
        raan=impactor_table["node_deg"].to_numpy(zero_copy_only=False),
        ap=impactor_table["argperi_deg"].to_numpy(zero_copy_only=False),
        M=impactor_table["M_deg"].to_numpy(zero_copy_only=False),
        time=Timestamp.from_mjd(
            impactor_table["epoch_mjd"].to_numpy(zero_copy_only=False), scale="tdb"
        ),
        origin=Origin.from_kwargs(
            code=np.full(len(impactor_table), "SUN", dtype="object")
        ),
        frame="ecliptic",
    )

    orbit = Orbits.from_kwargs(
        orbit_id=impactor_table["ObjID"].to_numpy(zero_copy_only=False),
        object_id=impactor_table["ObjID"].to_numpy(zero_copy_only=False),
        coordinates=keplerian_coords.to_cartesian(),
    )

    return orbit


def sorcha_output_to_od_observations(sorcha_output_file: str) -> Observations:
    """
    Convert Sorcha observations output files to Observations type.

    Parameters
    ----------
    sorcha_output_file : str
        Path to the Sorcha output file.

    Returns
    -------
    od_observations : qv.Table
        Observations object continaining the Sorcha observations.
        Returns None if the input file is empty.
    """
    logger.info(f"Reading Sorcha output file: {sorcha_output_file}")
    sorcha_observations_table = pa.csv.read_csv(sorcha_output_file)
    sort_indices = pc.sort_indices(
        sorcha_observations_table,
        sort_keys=[("ObjID", "ascending"), ("fieldMJD_TAI", "ascending")],
    )
    sorcha_observations_table = sorcha_observations_table.take(sort_indices)
    observations = Observations.empty()

    object_ids = pc.unique(sorcha_observations_table["ObjID"])

    for obj in object_ids:
        object_obs = sorcha_observations_table.filter(
            pc.equal(sorcha_observations_table["ObjID"], obj)
        )
        times = Timestamp.from_mjd(object_obs["fieldMJD_TAI"], scale="tai")
        times = times.rescale("utc")
        sigmas = np.full((len(object_obs), 6), np.nan)
        sigmas[:, 1] = object_obs["astrometricSigma_deg"].to_numpy(zero_copy_only=False)
        sigmas[:, 2] = object_obs["astrometricSigma_deg"].to_numpy(zero_copy_only=False)
        photometry = Photometry.from_kwargs(
            mag=object_obs["trailedSourceMag"],
            mag_sigma=object_obs["trailedSourceMagSigma"],
            filter=object_obs["optFilter"],
        )
        coordinates = SphericalCoordinates.from_kwargs(
            lon=object_obs["RA_deg"],
            lat=object_obs["Dec_deg"],
            origin=Origin.from_kwargs(
                code=pa.repeat("X05", len(object_obs)),
            ),
            time=times,
            frame="equatorial",
            covariance=CoordinateCovariances.from_sigmas(sigmas),
        )

        object_observation = Observations.from_kwargs(
            obs_id=[f"{obj}_{i}" for i in range(len(object_obs))],
            orbit_id=pa.repeat(obj, len(object_obs)),
            coordinates=coordinates,
            observers=Observers.from_codes(coordinates.origin.code, coordinates.time),
            photometry=photometry,
            linked=object_obs["object_linked"],
        )

        observations = qv.concatenate([observations, object_observation])

    return observations
