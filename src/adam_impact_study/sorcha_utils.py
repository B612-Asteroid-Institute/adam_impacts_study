import glob
import logging
import os
import subprocess

import pandas as pd
import quivr as qv
from adam_core.orbits import Orbits
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

from adam_impact_study.conversions import Observations, sorcha_output_to_od_observations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_config_file_timeframe(impact_date, config_file):
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

    pointing_command = f"SELECT observationId, observationStartMJD as observationStartMJD_TAI, visitTime, visitExposureTime, filter, seeingFwhmGeom as seeingFwhmGeom_arcsec, seeingFwhmEff as seeingFwhmEff_arcsec, fiveSigmaDepth as fieldFiveSigmaDepth_mag , fieldRA as fieldRA_deg, fieldDec as fieldDec_deg, rotSkyPos as fieldRotSkyPos_deg FROM observations WHERE observationStartMJD < {impact_date} ORDER BY observationId"
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
"""
    with open(config_file, "w") as f:
        f.write(config_text)
    return config_file


def generate_sorcha_orbits(adam_orbits: Orbits, sorcha_orbits_file: str) -> None:
    """
    Generate a Sorcha orbit file from a DataFrame of impactor data.

    Parameters
    ----------
    adam_orbits : `~adam_core.orbits.orbits.Orbits`
        ADAM Orbits object containing orbital parameters for the impactors.

    sorcha_orbits_file : str
        Path to the file where the Sorcha orbit data will be saved.

    Returns
    -------
    None
        The function writes the Sorcha orbit data to a file.
    """
    coord_kep = adam_orbits.coordinates.to_keplerian()
    sorcha_df = pd.DataFrame(
        {
            "ObjID": adam_orbits.object_id,
            "FORMAT": ["KEP"] * len(adam_orbits),
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


def generate_sorcha_physical_params(
    sorcha_physical_params_file: str, physical_params_df: pd.DataFrame
) -> None:
    """
    Generate a Sorcha physical parameters file from a DataFrame of physical parameters.

    Parameters
    ----------
    sorcha_physical_params_file : str
        Path to the file where the Sorcha physical parameters data will be saved.
    physical_params_df : pandas.DataFrame
        DataFrame containing physical parameters with appropriate columns.

    Returns
    -------
    None
        The function writes the Sorcha physical parameters data to a file.
    """
    physical_params_df.to_csv(sorcha_physical_params_file, index=False, sep=" ")
    return


def run_sorcha(
    adam_orbits: Orbits,
    sorcha_config_file: str,
    sorcha_orbits_file: str,
    sorcha_physical_params_file: str,
    pointing_file: str,
    sorcha_output_dir: str,
    sorcha_output_stem: str,
) -> Observations:
    """
    Run the Sorcha software to generate observational data based on input orbital and physical parameters.

    Parameters
    ----------
    adam_orbits : Orbits
        ADAM Orbits object containing orbital parameters for the impactors.
    sorcha_config_file : str
        Path to the Sorcha configuration file.
    sorcha_orbits_dir : str
        Path to the file where the Sorcha orbits are written as input.
    sorcha_physical_params_file : str
        Path to the file where the Sorcha physical parameters data will be saved.
    physical_params_df : pd.DataFrame
        DataFrame containing physical parameters for the impactors.
    pointing_file : str
        Path to the file containing pointing data.
    sorcha_output_file : str
        Name of the Sorcha output file.
    sorcha_output_stem : str
        File stem for the Sorcha output files.
    RESULT_DIR : str
        Directory where the results will be stored.

    Returns
    -------
    sorcha_observations : Observations (qv.Table)
        Observations object containing the Sorcha observations.
    """

    generate_sorcha_orbits(adam_orbits, sorcha_orbits_file)
    logger.info(f"Generated Sorcha orbits file: {sorcha_orbits_file}")

    # Get the output directory from the output file path
    os.makedirs(sorcha_output_dir, exist_ok=True)

    # Run Sorcha
    sorcha_command_string = (
        f"sorcha run -c {sorcha_config_file} -p {sorcha_physical_params_file} "
        f"--orbits {sorcha_orbits_file} --pointing-db {pointing_file} -o {sorcha_output_dir} "
        f"--stem {sorcha_output_stem} -f"
    )
    logger.info(f"Running Sorcha with command: {sorcha_command_string}")
    logger.info(f"Outputs will be saved to: {sorcha_output_dir}")
    # os.makedirs(f"{RESULT_DIR}/{sorcha_output_name}", exist_ok=True)

    subprocess.run(sorcha_command_string, shell=True)

    result_files = glob.glob(f"{sorcha_output_dir}/*.csv")

    if len(result_files) == 0:
        logger.warning(f"No output files found in {sorcha_output_dir}")
        return Observations.empty()

    # Read the sorcha output
    observations = Observations.empty()
    for result_file in result_files:
        observations = qv.concatenate([observations, sorcha_output_to_od_observations(result_file)])

    return observations
