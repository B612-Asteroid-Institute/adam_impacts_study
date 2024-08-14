import numpy as np
import pandas as pd
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
    Origin,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbit_determination import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.time import Timestamp


# Convert impactors data frame to adam orbit
def impactor_to_adam_orbit(impactor_df):
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


# Generate a dataframe from sorcha output
def sorcha_output_to_df(sorcha_output_file):
    sorcha_output_df = pd.read_csv(sorcha_output_file, float_precision="round_trip")
    # sort by object id
    sorcha_output_df = sorcha_output_df.sort_values(
        by=["ObjID", "fieldMJD_TAI"], ignore_index=True
    )
    return sorcha_output_df


# Covert sorcha orbit format to adam orbit (for testing of sorcha demo files)
def adam_orbit_from_sorcha_input(sorcha_orbits_file):
    sorcha_orbits_df = pd.read_csv(sorcha_orbits_file, sep=" ")
    keplerian_coords = KeplerianCoordinates.from_kwargs(
        a=sorcha_orbits_df["a"],
        e=sorcha_orbits_df["e"],
        i=sorcha_orbits_df["inc"],
        raan=sorcha_orbits_df["node"],
        ap=sorcha_orbits_df["argPeri"],
        M=sorcha_orbits_df["ma"],
        time=Timestamp.from_mjd(sorcha_orbits_df["epochMJD_TDB"].values, scale="tdb"),
        origin=Origin.from_kwargs(
            code=np.full(len(sorcha_orbits_df), "SUN", dtype="object")
        ),
        frame="ecliptic",
    )
    orbit = Orbits.from_kwargs(
        orbit_id=sorcha_orbits_df["ObjID"],
        object_id=sorcha_orbits_df["ObjID"],
        coordinates=keplerian_coords.to_cartesian(),
    )
    return orbit


def sorcha_output_to_od_observations(sorcha_observations_df):
    od_observations = {}

    object_ids = sorcha_observations_df["ObjID"].unique()

    for obj in object_ids:
        object_obs = sorcha_observations_df[sorcha_observations_df["ObjID"] == obj]
        times = Timestamp.from_mjd(object_obs["fieldMJD_TAI"].values, scale="tai")
        times = times.rescale("utc")
        sigmas = np.full((len(object_obs), 6), np.nan)
        sigmas[:, 1] = object_obs["astrometricSigma_deg"]
        sigmas[:, 2] = object_obs["astrometricSigma_deg"]
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
        od_observations[obj] = OrbitDeterminationObservations.from_kwargs(
            id=[f"{obj}_{i}" for i in range(len(object_obs))],
            coordinates=coordinates_sorted,
            observers=Observers.from_code("X05", coordinates_sorted.time),
        )

    return od_observations


def sorcha_df_to_fo_input(sorcha_df, fo_file_name):
    with open(fo_file_name, "w") as w:
        w.write("trkSub|stn|obsTime|ra|dec|rmsRA|rmsDec|mag|rmsMag|band\n")
        for index, row in sorcha_df.iterrows():
            time_tai = Timestamp.from_mjd([row["fieldMJD_TAI"]], scale="tai")
            time_utc = time_tai.rescale("utc")
            time = time_utc.to_astropy()
            w.write(
                f"{row['ObjID']}|X05|{time.isot[0]}|{row['RA_deg']}|"
                f"{row['Dec_deg']}|"
                f"{float(row['astrometricSigma_deg'])*3600}|"
                f"{float(row['astrometricSigma_deg'])*3600}|"
                f"{row['trailedSourceMag']}|"
                f"{row['trailedSourceMagSigma']}|"
                f"{row['optFilter']}\n"
            )
    return fo_file_name


# Convert the find_orb output to ADAM Orbit objects
def fo_to_adam_orbit_cov(elements_dict, covar_dict):
    orbits_dict = {}
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
        orbits = Orbits.from_kwargs(
            object_id=[object_id],
            orbit_id=[object_id],
            coordinates=cartesian_coordinates,
        )
        orbits_dict[object_id] = orbits
    return orbits_dict
