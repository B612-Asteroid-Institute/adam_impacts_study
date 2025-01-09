import glob
import logging
import os
import subprocess
from typing import Optional

import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

from adam_impact_study.conversions import sorcha_output_to_od_observations
from adam_impact_study.types import ImpactorOrbits, Observations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotometricProperties(qv.Table):
    orbit_id = qv.LargeStringColumn()
    H_mf = qv.Float64Column()
    u_mf = qv.Float64Column()
    g_mf = qv.Float64Column()
    i_mf = qv.Float64Column()
    z_mf = qv.Float64Column()
    y_mf = qv.Float64Column()
    GS = qv.Float64Column()


def remove_quotes(file_path: str) -> None:
    """
    Remove quotes from a file.

    Parameters
    ----------
    file_path : str
        Path to the file to remove quotes from.
    """
    temp_file_path = file_path + ".tmp"
    with open(file_path, "rb") as infile, open(temp_file_path, "wb") as outfile:
        while True:
            chunk = infile.read(65536)
            if not chunk:
                break
            chunk = chunk.replace(b'"', b"")
            outfile.write(chunk)
    os.replace(temp_file_path, file_path)


def write_config_file_timeframe(impact_date: Timestamp, config_file: str) -> str:
    """
    Write a Sorcha configuration file for a given impact date.

    Parameters
    ----------
    impact_date : float
        Impact date in MJD.
    config_file : str
        Path to the file where the Sorcha configuration data will be saved.

    Returns
    -------
    config_file : str
        Path to the file where the Sorcha configuration data was saved.
    """
    logger.info("Writing Sorcha config file")
    assist_planets = de440
    assist_small_bodies = de441_n16

    # Ensure impact_date is in TAI before serializing to mjd
    impact_date_tai = impact_date.rescale("tai")
    impact_date_mjd = impact_date_tai.mjd()[0]

    pointing_command = f"SELECT observationId, observationStartMJD as observationStartMJD_TAI, visitTime, visitExposureTime, filter, seeingFwhmGeom as seeingFwhmGeom_arcsec, seeingFwhmEff as seeingFwhmEff_arcsec, fiveSigmaDepth as fieldFiveSigmaDepth_mag , fieldRA as fieldRA_deg, fieldDec as fieldDec_deg, rotSkyPos as fieldRotSkyPos_deg FROM observations WHERE observationStartMJD < {impact_date_mjd} ORDER BY observationId"
    config_text = f"""
[Sorcha Configuration File]

[INPUT]
ephemerides_type = ar
eph_format = csv
size_serial_chunk = 5000
aux_format = whitespace
pointing_sql_query = {pointing_command}

[SIMULATION]
ar_ang_fov = 2.06
ar_fov_buffer = 0.2
ar_picket = 1
ar_obs_code = X05
ar_healpix_order = 6

[FILTERS]
observing_filters = r,g,i,z,u,y

[SATURATION]
bright_limit = 16.0

[PHASECURVES]
phase_function = HG

[FOV]
camera_model = footprint

[FADINGFUNCTION]
fading_function_on = True
fading_function_width = 0.1
fading_function_peak_efficiency = 1.

[LINKINGFILTER]
SSP_detection_efficiency = 0.95
SSP_number_observations = 2
SSP_separation_threshold = 0.5
SSP_maximum_time = 0.0625
SSP_number_tracklets = 3
SSP_track_window = 15
SSP_night_start_utc = 16.0

[OUTPUT]
output_format = csv
output_columns = basic
position_decimals = 7
magnitude_decimals = 3

[LIGHTCURVE]
lc_model = none

[ACTIVITY]
comet_activity = none
    
[AUXILIARY]
jpl_planets = {assist_planets}
jpl_small_bodies = {assist_small_bodies}

[EXPERT]
ar_use_integrate = True
"""
    with open(config_file, "w") as f:
        f.write(config_text)
    return config_file


def write_sorcha_orbits_file(orbits: Orbits, sorcha_orbits_file: str) -> None:
    """
    Generate a Sorcha orbit file from a DataFrame of impactor data.

    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits`
        ADAM core Orbits object containing orbital parameters for the impactors.

    sorcha_orbits_file : str
        Path to the file where the Sorcha orbit data will be saved.

    Returns
    -------
    None
        The function writes the Sorcha orbit data to a file.
    """
    coord_kep = orbits.coordinates.to_keplerian()
    sorcha_df = pd.DataFrame(
        {
            "ObjID": orbits.orbit_id,
            "FORMAT": ["KEP"] * len(orbits),
            "a": coord_kep.a,
            "e": coord_kep.e,
            "inc": coord_kep.i,
            "node": coord_kep.raan,
            "argPeri": coord_kep.ap,
            "ma": coord_kep.M,
            "epochMJD_TDB": coord_kep.time.mjd(),
        }
    )
    sorcha_df.to_csv(sorcha_orbits_file, index=False, sep=" ")
    return


def photometric_properties_to_sorcha_table(
    properties: PhotometricProperties, main_filter: str
) -> pa.Table:
    """
    Convert a PhotometricProperties table to a Sorcha table.

    Parameters
    ----------
    properties : `~adam_impact_study.physical_params.PhotometricProperties`
        Table containing the physical parameters of the impactors.
    main_filter : str
        Name of the main filter.

    Returns
    -------
    table : `pyarrow.Table`
        Table containing the physical parameters of the impactors.
    """
    table = properties.table
    column_names = table.column_names
    new_names = []
    for c in column_names:
        if c == "orbit_id":
            new_name = "ObjID"
        elif c.endswith("_mf"):
            new_name = (
                f"H_{main_filter}"
                if c == "H_mf"
                else c.replace("_mf", f"-{main_filter}")
            )
        else:
            new_name = c
        new_names.append(new_name)
    table = table.rename_columns(new_names)
    return table


def write_phys_params_file(
    photometric_properties: PhotometricProperties,
    properties_file: str,
    filter_band: str = "r",
) -> None:
    """
    Write the physical parameters to a file.

    Parameters
    ----------
    photometric_properties : `PhotometricProperties`
        Table containing the physical parameters of the impactors.
    properties_file : str
        Path to the file where the physical parameters will be saved.
    filter_band : str, optional
        Filter band to use for the photometric properties (default: "r").
    """
    properties_table = photometric_properties_to_sorcha_table(
        photometric_properties, filter_band
    )
    pa.csv.write_csv(
        properties_table,
        properties_file,
        write_options=pa.csv.WriteOptions(
            include_header=True,
            delimiter=" ",
            quoting_style="needed",
        ),
    )
    remove_quotes(properties_file)
    return


def run_sorcha(
    impactor_orbit: ImpactorOrbits,
    pointing_file: str,
    working_dir: str,
    seed: Optional[int] = None,
) -> Observations:
    """Run Sorcha with directory-based paths"""
    assert len(impactor_orbit) == 1, "Currently only one object is supported"
    # Generate input files
    orbits_file = os.path.join(working_dir, "orbits.csv")
    params_file = os.path.join(working_dir, "params.csv")
    config_file = os.path.join(working_dir, "config.ini")
    output_stem = "observations"

    impact_date = impactor_orbit.impact_time
    write_sorcha_orbits_file(impactor_orbit.orbits(), orbits_file)
    write_phys_params_file(
        impactor_orbit.photometric_properties(), params_file, filter_band="r"
    )
    write_config_file_timeframe(impact_date, config_file)

    # Run Sorcha to generate observational data
    sorcha_command = (
        f"SORCHA_SEED={seed} "
        f"sorcha run -c {config_file} -p {params_file} "
        f"--orbits {orbits_file} --pointing-db {pointing_file} "
        f"-o {working_dir} --stem {output_stem} -f"
    )

    logger.info(f"Running sorcha command: {sorcha_command}")

    subprocess.run(sorcha_command, shell=True)

    # Process results
    result_files = glob.glob(f"{working_dir}/{output_stem}*.csv")
    if not result_files:
        return Observations.empty()

    observations = Observations.empty()
    for result_file in result_files:
        observations = qv.concatenate(
            [observations, sorcha_output_to_od_observations(result_file)]
        )

    return observations
