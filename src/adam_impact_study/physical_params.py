import quivr as qv
import numpy as np
import pyarrow as pa
import os
import yaml

import unittest
from unittest.mock import patch

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
    r_mf = qv.Float64Column(default=0.0)
    i_mf = qv.Float64Column(default=0.0)
    z_mf = qv.Float64Column(default=0.0)
    y_mf = qv.Float64Column(default=0.0)
    GS = qv.Float64Column(default=0.15)


def photometric_properties_to_sorcha_table(
    properties: PhotometricProperties, main_filter: str
) -> pa.Table:
    table = properties.table

    # Map the columns to the correct filter. So all _mf colors
    # will be renamed to {filter}-{main_filter}. The absolute magnitude
    # will be renamed to H_{main_filter} (note an underscore instead of a dash)
    column_names = table.column_names
    new_names = []
    for c in column_names:
        if c.endswith("_mf"):
            if c == "H_mf":
                new_name = f"H_{main_filter}"
            else:
                new_name = c.replace("_mf", f"-{main_filter}")
        else:
            new_name = c

        new_names.append(new_name)

    table = table.rename_columns(new_names)
    return table


def remove_quotes(file_path: str) -> None:
    # Create a temporary file path
    temp_file_path = file_path + ".tmp"

    # Open the original file for reading in binary mode
    # and the temporary file for writing in binary mode
    with open(file_path, "rb") as infile, open(temp_file_path, "wb") as outfile:
        while True:
            # Read a chunk of data (e.g., 64KB)
            chunk = infile.read(65536)
            if not chunk:
                break
            # Remove quote characters
            chunk = chunk.replace(b'"', b"")
            # Write the processed chunk to the temporary file
            outfile.write(chunk)

    # Replace the original file with the temporary file
    os.replace(temp_file_path, file_path)


def write_phys_params_file(properties_table, properties_file):
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


def select_albedo_from_range(albedo_min, albedo_max, seed = 13612):
    rng = np.random.default_rng(seed)
    return rng.uniform(albedo_min, albedo_max)


def select_asteroid_size(min_diam, max_diam, seed = 13612):
    rng = np.random.default_rng(seed)
    return rng.uniform(min_diam, max_diam)


def determine_ast_class(percent_C, percent_S):
    assert percent_C + percent_S == 1, "Pencentage of C and S asteroids must equal 1"

    rng = np.random.default_rng()
    if rng.random() < percent_C:
        return "C"
    else:
        return "S"
    
def calculate_H(diameter, albedo):
    return 15.618 - 5 * np.log10(diameter) - 2.5 * np.log10(albedo)
    
def read_config_file(config_file):
    with open(config_file, "r") as f:
        config_str = f.read()
    return ImpactStudyConfig.from_kwargs(

    )
    
def load_config(file_path: str) -> ImpactStudyConfig:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    config = ImpactStudyConfig(**config_data)
    return config


def create_physical_params(config_file):
    config = load_config(config_file)
    phys_params = PhotometricProperties.empty()
    for ast in range(config.n_asteroids):
        ast_class = determine_ast_class()
        d = select_asteroid_size(config.min_diam, config.max_diam)
        
        if ast_class == "C":
            albedo = select_albedo_from_range(config.C_albedo_min, config.C_albedo_max)
            H = calculate_H(d, albedo)
            phy_param = PhotometricProperties.from_kwargs(
                H_mf=[H],
                u_mf=[config.u_r_C],
                g_mf=[config.g_r_C],
                r_mf=[config.i_r_C],
                i_mf=[config.z_r_C],
                z_mf=[config.y_r_C],
                GS=[0.15],
                ObjID=[str(ast)],
            )
        elif ast_class == "S":
            albedo = select_albedo_from_range(config.S_albedo_min, config.S_albedo_max)
            H = calculate_H(d, albedo)
            phy_param = PhotometricProperties.from_kwargs(
                H_mf=[H],
                u_mf=[config.u_r_S],
                g_mf=[config.g_r_S],
                r_mf=[config.i_r_S],
                i_mf=[config.z_r_S],
                z_mf=[config.y_r_S],
                GS=[0.15],
                ObjID=[str(ast)],
            )
        
        phys_params = qv.concatenate([phys_params, phy_param])

    return phys_params