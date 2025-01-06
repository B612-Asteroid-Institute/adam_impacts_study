import glob
import logging
import os
import subprocess
from typing import Optional

import pandas as pd
import quivr as qv
from adam_core.orbits import Orbits
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

from adam_impact_study.conversions import sorcha_output_to_od_observations
from adam_impact_study.physical_params import (
    create_physical_params_single,
    photometric_properties_to_sorcha_table,
    write_phys_params_file,
)
from adam_impact_study.types import Observations

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

[EXPERT]
ar_use_integrate = True
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


def generate_physical_params_file(
    population_config_file: str,
    object_id: str,
    output_file: str,
    filter_band: str = "r",
    seed: int = 13612,
) -> None:
    """
    Generate a physical parameters file for Sorcha from a population config file.

    Parameters
    ----------
    population_config_file : str
        Path to the population configuration file.
    object_id : str
        Object identifier to generate parameters for.
    output_file : str
        Path where the physical parameters file should be written.
    filter_band : str, optional
        Filter band to use for photometric properties (default: "r").

    Returns
    -------
    None
        The function writes the physical parameters to the specified output file.
    """
    phys_params = create_physical_params_single(population_config_file, object_id, seed)
    phys_para_file_str = photometric_properties_to_sorcha_table(
        phys_params, filter_band
    )
    write_phys_params_file(phys_para_file_str, output_file)


def run_sorcha(
    adam_orbits: Orbits,
    pointing_file: str,
    population_config_file: str,
    working_dir: str,
    seed: Optional[int] = None,
) -> Observations:
    """Run Sorcha with directory-based paths"""
    assert len(adam_orbits) == 1, "Currently only one object is supported"
    # Generate input files
    orbits_file = os.path.join(working_dir, "orbits.csv")
    params_file = os.path.join(working_dir, "params.csv")
    config_file = os.path.join(working_dir, "config.ini")
    output_stem = "observations"

    generate_sorcha_orbits(adam_orbits, orbits_file)

    generate_physical_params_file(
        population_config_file, adam_orbits.object_id[0].as_py(), params_file, seed=seed
    )

    impact_date = adam_orbits.coordinates.time.add_days(30)

    write_config_file_timeframe(impact_date.mjd()[0], config_file)

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
