import glob
import logging
import os
import pathlib
import subprocess
from typing import Optional

import pandas as pd
import pooch
import pyarrow as pa
import quivr as qv
from adam_core.observers.utils import calculate_observing_night
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import _de441_n16_md5, de441_n16
from mpc_obscodes import _mpc_obscodes_md5, mpc_obscodes
from naif_de440 import _de440_md5, de440
from naif_eop_high_prec import _eop_high_prec_md5, eop_high_prec
from naif_eop_historical import _eop_historical_md5, eop_historical
from naif_eop_predict import _eop_predict_md5, eop_predict
from naif_leapseconds import _leapseconds_md5, leapseconds

from adam_impact_study.conversions import sorcha_output_to_od_observations
from adam_impact_study.types import ImpactorOrbits, Observations, PhotometricProperties

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))


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


def write_config_file_timeframe(
    config_file: str,
    impact_date: Timestamp,
    assist_epsilon: float,
    assist_min_dt: float,
    assist_initial_dt: float,
    assist_adaptive_mode: int,
) -> str:
    """
    Write a Sorcha configuration file for a given impact date.

    Parameters
    ----------
    config_file : str
        Path to the file where the Sorcha configuration data will be saved.
    impact_date : float
        Impact date in MJD.
    assist_epsilon : float
        Epsilon value for ASSIST
    assist_min_dt : float
        Minimum time step for ASSIST
    assist_initial_dt : float
        Initial time step for ASSIST
    assist_adaptive_mode : int
        Adaptive mode for ASSIST

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
    # impact_date_mjd = impact_date_tai.mjd()[0]
    # Go past impact date for testing
    impact_date_mjd = impact_date_tai.add_days(1).mjd()[0]

    # get the parent directory of the config file, with pathlib
    config_dir = pathlib.Path(config_file).parent
    meta_kernel_file = config_dir / "meta_kernel.txt"

    sorcha_cache_dir = pooch.os_cache("sorcha")
    site_packages_lib_dir = pathlib.Path(qv.__file__).parent.parent
    # grab the last two elements of the path
    kernels_to_load = [
        f"'$B/{pathlib.Path(leapseconds).relative_to(site_packages_lib_dir)}'",
        f"'$B/{pathlib.Path(eop_historical).relative_to(site_packages_lib_dir)}'",
        f"'$B/{pathlib.Path(eop_predict).relative_to(site_packages_lib_dir)}'",
        "'$A/pck00010.pck'",
        f"'$B/{pathlib.Path(de440).relative_to(site_packages_lib_dir)}'",
        f"'$B/{pathlib.Path(eop_high_prec).relative_to(site_packages_lib_dir)}'",
    ]
    # Manually create our meta_kernel.txt file
    meta_kernel_txt = f"""\\begindata
PATH_VALUES=(
'{sorcha_cache_dir}',
'{site_packages_lib_dir}',
)
PATH_SYMBOLS=(
'A',
'B',
)

KERNELS_TO_LOAD=(
"""
    for kernel in kernels_to_load:
        meta_kernel_txt += f"{kernel},\n"
    meta_kernel_txt += """
)

\\begintext
"""

    with open(meta_kernel_file, "w") as f:
        f.write(meta_kernel_txt)

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
# NOTE: Detection efficiency changed from LSST design specification
# to show theoretical optimal performance.
SSP_detection_efficiency = 1.0 
SSP_number_observations = 2
SSP_separation_threshold = 0.5
SSP_maximum_time = 0.0625
SSP_number_tracklets = 3
SSP_track_window = 15
SSP_night_start_utc = 16.0
drop_unlinked = False

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
# jpl_planets_version = {pathlib.Path(_de440_md5).read_text()}
jpl_small_bodies = {assist_small_bodies}
# jpl_small_bodies_version = {pathlib.Path(_de441_n16_md5).read_text()}
planet_ephemeris = {assist_planets}
# planet_ephemeris_version = {pathlib.Path(_de440_md5).read_text()}
earth_predict = {eop_predict}
# earth_predict_version = {pathlib.Path(_eop_predict_md5).read_text()}
earth_historical = {eop_historical}
# earth_historical_version = {pathlib.Path(_eop_historical_md5).read_text()}
earth_high_precision = {eop_high_prec}
# earth_high_precision_version = {pathlib.Path(_eop_high_prec_md5).read_text()}
observatory_codes = {mpc_obscodes}
# observatory_codes_version = {pathlib.Path(_mpc_obscodes_md5).read_text()}
leap_seconds = {leapseconds}
# leap_seconds_version = {pathlib.Path(_leapseconds_md5).read_text()}
meta_kernel = {str(meta_kernel_file.absolute())}

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
    simulation_end_date: Timestamp,
    pointing_file: str,
    working_dir: str,
    assist_epsilon: float,
    assist_min_dt: float,
    assist_initial_dt: float,
    assist_adaptive_mode: int,
    seed: Optional[int] = None,
) -> Observations:
    """Run Sorcha with directory-based paths

    Parameters
    ----------
    impactor_orbit : `~adam_impact_study.types.ImpactorOrbits`
        Orbit of the impactor.
    simulation_end_date : `~adam_core.time.Timestamp`
        End date of the simulation. Generally this is impact_date - 1 day to avoid problems with
        propagation of hyperbolic orbits in sorcha.
    pointing_file : str
        Path to the sorcha pointing database file. This will determine the start date of the simulation.
    working_dir : str
        Path to the working directory.
    assist_epsilon : float
        Epsilon value for ASSIST
    assist_min_dt : float
        Minimum time step for ASSIST
    assist_initial_dt : float
        Initial time step for ASSIST
    assist_adaptive_mode : int
        Adaptive mode for ASSIST
    """
    assert len(impactor_orbit) == 1, "Currently only one object is supported"
    # Generate input files
    orbits_file = os.path.join(working_dir, "orbits.csv")
    params_file = os.path.join(working_dir, "params.csv")
    config_file = os.path.join(working_dir, "config.ini")
    output_stem = "observations"

    write_sorcha_orbits_file(impactor_orbit.orbits(), orbits_file)
    write_phys_params_file(
        impactor_orbit.photometric_properties(), params_file, filter_band="r"
    )

    # TODO: Investigate if we can avoid limiting the observation time span
    # to 1 day prior to impact to avoid edge cases where the propagation
    # in sorcha approaches singularity-like behavior
    write_config_file_timeframe(
        config_file,
        simulation_end_date,
        assist_epsilon,
        assist_min_dt,
        assist_initial_dt,
        assist_adaptive_mode,
    )

    # Run Sorcha to generate observational data
    sorcha_command = (
        f"SORCHA_SEED={seed} "
        f"sorcha run -c {config_file} -p {params_file} "
        f"--orbits {orbits_file} --pointing-db {pointing_file} "
        f"-o {working_dir} --stem {output_stem} -f"
    )

    logger.info(f"Running sorcha for {impactor_orbit.orbit_id[0].as_py()}")
    logger.debug(f"Running sorcha command: {sorcha_command}")

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

    # This is not strictly sorcha related, but we definitely want observing
    # night everywhere downstream of this point, so we do it here.
    # Also, sorting is a good idea.
    observations = observations.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos", "coordinates.origin.code"]
    )
    # Add the observing night column to the observations
    observations = observations.set_column(
        "observing_night",
        calculate_observing_night(
            observations.coordinates.origin.code, observations.coordinates.time
        ),
    )

    return observations
