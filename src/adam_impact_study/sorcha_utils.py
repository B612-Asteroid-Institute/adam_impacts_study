import os
import subprocess

import pandas as pd
import quivr as qv
from adam_impact_study.conversions import sorcha_output_to_df


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


# Generate sorcha orbit files from a dataframe
def generate_sorcha_orbits(impactors_df, sorcha_orbits_file):
    """
    Generate a Sorcha orbit file from a DataFrame of impactor data.

    Parameters
    ----------
    impactors_df : pandas.DataFrame
        DataFrame containing orbital parameters for impactors with columns:
        - "ObjID": Object ID
        - "a_au": Semi-major axis (au)
        - "e": Eccentricity
        - "i_deg": Inclination (degrees)
        - "node_deg": Longitude of the ascending node (degrees)
        - "argperi_deg": Argument of periapsis (degrees)
        - "M_deg": Mean anomaly (degrees)
        - "epoch_mjd": Epoch in Modified Julian Date (TDB)
    sorcha_orbits_file : str
        Path to the file where the Sorcha orbit data will be saved.

    Returns
    -------
    None
        The function writes the Sorcha orbit data to a file.
    """
    sorcha_df = pd.DataFrame(
        {
            "ObjID": impactors_df["ObjID"],
            "FORMAT": ["KEP"] * len(impactors_df),  # Use 'KEP' for all rows
            "a": impactors_df["a_au"],
            "e": impactors_df["e"],
            "inc": impactors_df["i_deg"],
            "node": impactors_df["node_deg"],
            "argPeri": impactors_df["argperi_deg"],
            "ma": impactors_df["M_deg"],
            "epochMJD_TDB": impactors_df["epoch_mjd"],
        }
    )
    sorcha_df.to_csv(sorcha_orbits_file, index=False, sep=" ")
    return


# Generate physical params files from a dataframe
def generate_sorcha_physical_params(sorcha_physical_params_file, physical_params_df):
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
    impactor_df,
    sorcha_config_file,
    sorcha_orbits_file,
    sorcha_physical_params_file,
    sorcha_output_file,
    physical_params_df,
    pointing_file,
    sorcha_output_name,
    RESULT_DIR,
):
    """
    Run the Sorcha software to generate observational data based on input orbital and physical parameters.

    Parameters
    ----------
    impactor_df : pandas.DataFrame
        DataFrame containing orbital parameters for the impactors.
    sorcha_config_file : str
        Path to the Sorcha configuration file.
    sorcha_orbits_file : str
        Path to the file where the Sorcha orbits are written as input.
    sorcha_physical_params_file : str
        Path to the file where the Sorcha physical parameters data will be saved.
    sorcha_output_file : str
        Name of the Sorcha output file.
    physical_params_df : pandas.DataFrame
        DataFrame containing physical parameters for the impactors.
    pointing_file : str
        Path to the file containing pointing data.
    sorcha_output_name : str
        Name for the output directory where Sorcha results will be saved.
    RESULT_DIR : str
        Directory where the results will be stored.

    Returns
    -------
    sorcha_observations_df : pandas.DataFrame
        DataFrame containing the Sorcha-generated observations.
    """
    # Generate the sorcha input files
    generate_sorcha_orbits(impactor_df, sorcha_orbits_file)
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
    sorcha_observations_df = sorcha_output_to_df(
        f"{RESULT_DIR}/{sorcha_output_name}/{sorcha_output_file}"
    )
    return sorcha_observations_df
