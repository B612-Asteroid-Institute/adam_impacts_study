import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
    Origin,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp


class Photometry(qv.Table):
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn()


class Observations(qv.Table):
    obs_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()
    photometry = Photometry.as_column(nullable=True)


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
    impactor_df = pd.read_csv(impactor_file, float_precision="round_trip")
    keplerian_coords = KeplerianCoordinates.from_kwargs(
        a=impactor_df["a_au"],
        e=impactor_df["e"],
        i=impactor_df["i_deg"],
        raan=impactor_df["node_deg"],
        ap=impactor_df["argperi_deg"],
        M=impactor_df["M_deg"],
        time=Timestamp.from_mjd(impactor_df["epoch_mjd"].values, scale="tdb"),
        origin=Origin.from_kwargs(
            code=np.full(len(impactor_df), "SUN", dtype="object")
        ),
        frame="ecliptic",
    )
    orbit = Orbits.from_kwargs(
        orbit_id=impactor_df["ObjID"],
        object_id=impactor_df["ObjID"],
        coordinates=keplerian_coords.to_cartesian(),
    )
    return orbit


def sorcha_output_to_od_observations(sorcha_output_file: str) -> Optional[Observations]:
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

    sorcha_observations_df = pd.read_csv(
        sorcha_output_file, float_precision="round_trip"
    )
    sorcha_observations_df = sorcha_observations_df.sort_values(
        by=["ObjID", "fieldMJD_TAI"], ignore_index=True
    )
    od_observations = None

    object_ids = sorcha_observations_df["ObjID"].unique()

    for obj in object_ids:
        object_obs = sorcha_observations_df[sorcha_observations_df["ObjID"] == obj]
        times = Timestamp.from_mjd(object_obs["fieldMJD_TAI"].values, scale="tai")
        times = times.rescale("utc")
        sigmas = np.full((len(object_obs), 6), np.nan)
        sigmas[:, 1] = object_obs["astrometricSigma_deg"]
        sigmas[:, 2] = object_obs["astrometricSigma_deg"]
        photometry = Photometry.from_kwargs(
            mag=object_obs["trailedSourceMag"],
            mag_sigma=object_obs["trailedSourceMagSigma"],
            filter=object_obs["optFilter"],
        )
        coordinates = SphericalCoordinates.from_kwargs(
            lon=object_obs["RA_deg"],
            lat=object_obs["Dec_deg"],
            origin=Origin.from_kwargs(
                code=np.full(len(object_obs), "X05", dtype="object")
            ),
            time=times,
            frame="equatorial",
            covariance=CoordinateCovariances.from_sigmas(sigmas),
        )
        coordinates_sorted = coordinates.sort_by(
            [
                ("time.days", "ascending"),
                ("time.nanos", "ascending"),
                ("origin.code", "ascending"),
            ]
        )

        od_observation = Observations.from_kwargs(
            obs_id=[f"{obj}_{i}" for i in range(len(object_obs))],
            object_id=pa.repeat(obj, len(object_obs)),
            coordinates=coordinates_sorted,
            observers=Observers.from_code("X05", coordinates_sorted.time),
            photometry=photometry,
        )

        if od_observations is None:
            od_observations = od_observation
        else:
            od_observations = qv.concatenate([od_observations, od_observation])

    return od_observations


def od_observations_to_fo_input(
    od_observations: Observations, fo_file_name: str
) -> str:
    """
    Convert an Observations object into a Find_Orb input file.

    Parameters
    ----------
    od_observations : qv.Table
        Observations object containing observations to be converted.

    fo_file_name : str
        Name of the Find_Orb input file to be created.

    object_id : str
        Object ID of orbit connected to observations.

    Returns
    -------
    fo_file_name : str
        Path to the generated Find_Orb input file.
    """
    with open(fo_file_name, "w") as w:
        w.write("trkSub|stn|obsTime|ra|dec|rmsRA|rmsDec\n")  # |mag|rmsMag|band
        for obs in od_observations:
            sigmas = obs.coordinates.covariance.sigmas
            time_utc = obs.coordinates.time
            time = time_utc.to_astropy()
            w.write(
                f"{obs.object_id}|X05|{time.isot[0]}|{obs.coordinates.lon[0]}|"
                f"{obs.coordinates.lat[0]}|"
                f"{format(sigmas[0][1]*3600, '.5f')}|"
                f"{format(sigmas[0][2]*3600, '.5f')}|"
                f"{obs.photometry.mag}|"
                f"{obs.photometry.mag_sigma}|"
                f"{obs.photometry.filter}\n"
            )
    return fo_file_name


def read_fo_output(fo_output_dir: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Read the find_orb output files from the specified directory into dictionaries
    containing orbital elements and covariances.

    Parameters
    ----------
    fo_output_dir : str
        Directory path where find_orb output files (e.g., total.json and covar.json) are located.

    Returns
    -------
    elements_dict : dict
        Dictionary containing orbital elements for each object.
    covar_dict : dict
        Dictionary containing covariance matrices for each object.
    """
    covar_dict = read_fo_covariance(f"{fo_output_dir}/covar.json")
    elements_dict = read_fo_orbits(f"{fo_output_dir}/total.json")
    return elements_dict, covar_dict


def read_fo_covariance(covar_file: str) -> Dict[str, dict]:
    """
    Read the find_orb covariance JSON file into a dictionary.

    Parameters
    ----------
    covar_file : str
        Path to the find_orb covariance JSON file (covar.json).

    Returns
    -------
    covar_json : dict
        Dictionary containing the covariance data from the JSON file.
    """
    with open(covar_file, "r") as f:
        covar_json = json.load(f)
    return covar_json


def read_fo_orbits(input_file: str) -> Dict[str, dict]:
    """
    Read the find_orb total.json file into a dictionary of orbital elements.

    Parameters
    ----------
    input_file : str
        Path to the find_orb total.json file.

    Returns
    -------
    elements_dict : dict
        Dictionary containing orbital elements for each object.
    """
    with open(input_file, "r") as f:
        total_json = json.load(f)
    objects = total_json.get("objects", {})
    elements_dict = {}
    for object_id, object_data in objects.items():
        elements = object_data.get("elements", {})
        elements_dict[object_id] = elements
    return elements_dict


def fo_to_adam_orbit_cov(fo_output_folder: str) -> Orbits:
    """
    Convert Find_Orb output to ADAM Orbit objects, including covariance.

    Parameters
    ----------
    fo_output_folder : str
        Path to the folder containing Find_Orb output files.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        ADAM Orbit object created from the Find_Orb output data.
    """

    elements_dict, covar_dict = read_fo_output(fo_output_folder)

    orbits = None
    for object_id, elements in elements_dict.items():

        covar_matrix = np.array([covar_dict["covar"]])
        covar_state_vector = [covar_dict["state_vect"]]

        covariances_cartesian = CoordinateCovariances.from_matrix(covar_matrix)
        times = Timestamp.from_jd([covar_dict["epoch"]], scale="tdb")

        cartesian_coordinates = CartesianCoordinates.from_kwargs(
            x=[covar_state_vector[0][0]],
            y=[covar_state_vector[0][1]],
            z=[covar_state_vector[0][2]],
            vx=[covar_state_vector[0][3]],
            vy=[covar_state_vector[0][4]],
            vz=[covar_state_vector[0][5]],
            time=times,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=covariances_cartesian,
        )
        orbit = Orbits.from_kwargs(
            orbit_id=[object_id],
            object_id=[object_id],
            coordinates=cartesian_coordinates,
        )
        if orbits is None:
            orbits = orbit
        else:
            orbits = qv.concatenate([orbits, orbit])

    return orbits
