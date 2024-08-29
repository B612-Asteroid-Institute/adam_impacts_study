import os
import subprocess
from typing import Optional

import pandas as pd
import quivr as qv
from adam_core.orbits import Orbits

from adam_impact_study.conversions import Observations, sorcha_output_to_od_observations


class SorchaPhysicalParameters(qv.Table):

    ObjID = qv.LargeStringColumn()
    H_r = qv.Float64Column()
    ur = qv.Float64Column()
    gr = qv.Float64Column()
    ir = qv.Float64Column()
    zr = qv.Float64Column()
    yr = qv.Float64Column()
    GS = qv.Float64Column()


class SorchaOrbits(qv.Table):

    ObjID = qv.LargeStringColumn()
    FORMAT = qv.StringColumn()
    a = qv.Float64Column()
    e = qv.Float64Column()
    inc = qv.Float64Column()
    node = qv.Float64Column()
    argPeri = qv.Float64Column()
    ma = qv.Float64Column()
    epochMJD_TDB = qv.Float64Column()


class SorchaObservations(qv.Table):

    ObjID = qv.LargeStringColumn()
    fieldMJD_TAI = qv.Float64Column()
    fieldRA_deg = qv.Float64Column()
    fieldDec_deg = qv.Float64Column()
    RA_deg = qv.Float64Column()
    Dec_deg = qv.Float64Column()
    astrometricSigma_deg = qv.Float64Column()
    optFilter = qv.StringColumn()
    trailedSourceMag = qv.Float64Column()
    trailedSourceMagSigma = qv.Float64Column()
    fiveSigmaDepth_mag = qv.Float64Column()
    phase_deg = qv.Float64Column()
    Range_LTC_km = qv.Float64Column()
    RangeRate_LTC_km_s = qv.Float64Column()
    Obj_Sun_LTC_km = qv.Float64Column()


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
    sorcha_output_file: str,
    physical_params_df: pd.DataFrame,
    pointing_file: str,
    sorcha_output_name: str,
    RESULT_DIR: str,
) -> Optional[Observations]:
    """
    Run the Sorcha software to generate observational data based on input orbital and physical parameters.

    Parameters
    ----------
    adam_orbits : Orbits
        ADAM Orbits object containing orbital parameters for the impactors.
    sorcha_config_file : str
        Path to the Sorcha configuration file.
    sorcha_orbits_file : str
        Path to the file where the Sorcha orbits are written as input.
    sorcha_physical_params_file : str
        Path to the file where the Sorcha physical parameters data will be saved.
    sorcha_output_file : str
        Name of the Sorcha output file.
    physical_params_df : pd.DataFrame
        DataFrame containing physical parameters for the impactors.
    pointing_file : str
        Path to the file containing pointing data.
    sorcha_output_name : str
        Name for the output directory where Sorcha results will be saved.
    RESULT_DIR : str
        Directory where the results will be stored.

    Returns
    -------
    sorcha_observations : Observations (qv.Table)
        Observations object containing the Sorcha observations.
        Returns None if the input file is empty.
    """
    # Generate the sorcha input files
    generate_sorcha_orbits(adam_orbits, sorcha_orbits_file)
    generate_sorcha_physical_params(sorcha_physical_params_file, physical_params_df)

    # Run Sorcha
    sorcha_command_string = (
        f"sorcha run -c {sorcha_config_file} -p {sorcha_physical_params_file} "
        f"-ob {sorcha_orbits_file} -pd {pointing_file} -o {RESULT_DIR}/{sorcha_output_name} "
        f"-t {sorcha_output_name} -f"
    )
    print(f"Running Sorcha with command: {sorcha_command_string}")
    os.makedirs(f"{RESULT_DIR}/{sorcha_output_name}", exist_ok=True)
    subprocess.run(sorcha_command_string, shell=True)

    # Read the sorcha output
    sorcha_output_file = f"{RESULT_DIR}/{sorcha_output_name}/{sorcha_output_file}"
    od_observations = sorcha_output_to_od_observations(sorcha_output_file)
    return od_observations
