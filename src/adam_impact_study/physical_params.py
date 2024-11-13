import json
import os

import numpy as np
import pyarrow as pa
import quivr as qv


class ImpactStudyConfig(qv.Table):
    C_albedo_min = qv.Float64Column(default=0.03)
    C_albedo_max = qv.Float64Column(default=0.09)
    S_albedo_min = qv.Float64Column(default=0.10)
    S_albedo_max = qv.Float64Column(default=0.22)
    percent_C = qv.Float64Column(default=0.5)
    percent_S = qv.Float64Column(default=0.5)
    min_diam = qv.Float64Column(default=0.001)
    max_diam = qv.Float64Column(default=100)
    n_asteroids = qv.Int64Column(default=1000)
    u_r_C = qv.Float64Column(default=1.786)
    g_r_C = qv.Float64Column(default=0.474)
    i_r_C = qv.Float64Column(default=-0.119)
    z_r_C = qv.Float64Column(default=-0.126)
    y_r_C = qv.Float64Column(default=-0.131)
    u_r_S = qv.Float64Column(default=2.182)
    g_r_S = qv.Float64Column(default=0.65)
    i_r_S = qv.Float64Column(default=-0.2)
    z_r_S = qv.Float64Column(default=-0.146)
    y_r_S = qv.Float64Column(default=-0.151)


class PhotometricProperties(qv.Table):
    ObjID = qv.LargeStringColumn()
    H_mf = qv.Float64Column(default=0.0)
    u_mf = qv.Float64Column(default=0.0)
    g_mf = qv.Float64Column(default=0.0)
    i_mf = qv.Float64Column(default=0.0)
    z_mf = qv.Float64Column(default=0.0)
    y_mf = qv.Float64Column(default=0.0)
    GS = qv.Float64Column(default=0.15)


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
        if c.endswith("_mf"):
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


def write_phys_params_file(properties_table: pa.Table, properties_file: str) -> None:
    """
    Write the physical parameters to a file.

    Parameters
    ----------
    properties_table : `pyarrow.Table`
        Table containing the physical parameters of the impactors.
    properties_file : str
        Path to the file where the physical parameters will be saved.
    """
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


def select_albedo_from_range(
    albedo_min: float, albedo_max: float, seed: int = 13612
) -> float:
    """
    Select an albedo from a range.

    Parameters
    ----------
    albedo_min : float
        Minimum albedo value.
    albedo_max : float
        Maximum albedo value.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    albedo : float
        Albedo value.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(albedo_min, albedo_max)


def select_asteroid_size(min_diam: float, max_diam: float, seed: int = 13612) -> float:
    """
    Select an asteroid size from a range.

    Parameters
    ----------
    min_diam : float
        Minimum asteroid diameter, in kilometers.
    max_diam : float
        Maximum asteroid diameter, in kilometers.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    diam : float
        Asteroid diameter, in kilometers. 
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(min_diam, max_diam)


def determine_ast_class(percent_C: float, percent_S: float) -> str:
    """
    Determine the asteroid class based on the percentage of C and S asteroids.

    Parameters
    ----------
    percent_C : float
        Percentage of C asteroids.
    percent_S : float
        Percentage of S asteroids.

    Returns
    -------
    ast_class : str
        Asteroid class.
    """
    assert percent_C + percent_S == 1, "Percentage of C and S asteroids must equal 1"
    rng = np.random.default_rng()
    return "C" if rng.random() < percent_C else "S"


def calculate_H(diameter: float, albedo: float) -> float:
    """
    Calculate the absolute magnitude of an asteroid.

    Parameters
    ----------
    diameter : float
        Asteroid diameter, in kilometers.
    albedo : float
        Asteroid albedo.

    Returns
    -------
    H : float
        Absolute magnitude.
    """
    return 15.618 - 5 * np.log10(diameter) - 2.5 * np.log10(albedo)


def load_config(file_path: str) -> ImpactStudyConfig:
    """
    Load the impact study configuration from a file.

    Parameters
    ----------
    file_path : str
        Path to the configuration file.

    Returns
    -------
    config : `~adam_impact_study.physical_params.ImpactStudyConfig`
        Configuration object.
    """
    with open(file_path, "r") as file:
        config_data = json.load(file)
    config = ImpactStudyConfig.from_kwargs(
        C_albedo_min=[config_data.get("C_albedo_min", 0.03)],
        C_albedo_max=[config_data.get("C_albedo_max", 0.09)],
        S_albedo_min=[config_data.get("S_albedo_min", 0.10)],
        S_albedo_max=[config_data.get("S_albedo_max", 0.22)],
        percent_C=[config_data.get("percent_C", 0.5)],
        percent_S=[config_data.get("percent_S", 0.5)],
        min_diam=[config_data.get("min_diam", 0.01)],
        max_diam=[config_data.get("max_diam", 1)],
        n_asteroids=[config_data.get("n_asteroids", 1000)],
        u_r_C=[config_data.get("u_r_C", 1.786)],
        g_r_C=[config_data.get("g_r_C", 0.474)],
        i_r_C=[config_data.get("i_r_C", -0.119)],
        z_r_C=[config_data.get("z_r_C", -0.126)],
        y_r_C=[config_data.get("y_r_C", -0.131)],
        u_r_S=[config_data.get("u_r_S", 2.182)],
        g_r_S=[config_data.get("g_r_S", 0.65)],
        i_r_S=[config_data.get("i_r_S", -0.2)],
        z_r_S=[config_data.get("z_r_S", -0.146)],
        y_r_S=[config_data.get("y_r_S", -0.151)],
    )
    return config


def create_physical_params_single(
    config_file: str, obj_id: str
) -> PhotometricProperties:
    """
    Create physical parameters for a single impactor.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    obj_id : str
        Object ID of the impactor.

    Returns
    -------
    phys_params : `~adam_impact_study.physical_params.PhotometricProperties`
        Physical parameters of the impactor.
    """
    config = load_config(config_file)
    ast_class = determine_ast_class(
        config.percent_C.to_numpy()[0], config.percent_S.to_numpy()[0]
    )
    d = select_asteroid_size(
        config.min_diam.to_numpy()[0], config.max_diam.to_numpy()[0]
    )

    if ast_class == "C":
        albedo = select_albedo_from_range(
            config.C_albedo_min.to_numpy()[0], config.C_albedo_max.to_numpy()[0]
        )
        H = calculate_H(d, albedo)
        phys_params = PhotometricProperties.from_kwargs(
            H_mf=[H],
            u_mf=config.u_r_C,
            g_mf=config.g_r_C,
            i_mf=config.i_r_C,
            z_mf=config.z_r_C,
            y_mf=config.y_r_C,
            GS=[0.15],
            ObjID=[obj_id],
        )
    elif ast_class == "S":
        albedo = select_albedo_from_range(
            config.S_albedo_min.to_numpy()[0], config.S_albedo_max.to_numpy()[0]
        )
        H = calculate_H(d, albedo)
        phys_params = PhotometricProperties.from_kwargs(
            H_mf=[H],
            u_mf=config.u_r_S,
            g_mf=config.g_r_S,
            i_mf=config.i_r_S,
            z_mf=config.z_r_S,
            y_mf=config.y_r_S,
            GS=[0.15],
            ObjID=[obj_id],
        )
    return phys_params
