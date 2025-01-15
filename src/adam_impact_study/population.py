import json
import logging
import os
from typing import List

import numpy as np
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from .types import ImpactorOrbits
from .utils import seed_from_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopulationConfig(qv.Table):
    config_id = qv.LargeStringColumn()
    ast_class = qv.LargeStringColumn()
    albedo_min = qv.Float64Column()
    albedo_max = qv.Float64Column()
    percentage = qv.Float64Column()
    u_r = qv.Float64Column()
    g_r = qv.Float64Column()
    i_r = qv.Float64Column()
    z_r = qv.Float64Column()
    y_r = qv.Float64Column()

    @classmethod
    def from_json(cls, json_file: str) -> "PopulationConfig":
        with open(json_file, "r") as file:
            config_data = json.load(file)
        return PopulationConfig.from_kwargs(**config_data)

    def to_json(self, json_file: str) -> None:
        with open(json_file, "w") as file:
            json.dump(self.table.to_pydict(), file, indent=4)

    @classmethod
    def default(cls) -> "PopulationConfig":
        C_type = PopulationConfig.from_kwargs(
            config_id=["default"],
            ast_class=["C"],
            albedo_min=[0.03],
            albedo_max=[0.09],
            percentage=[0.5],
            u_r=[1.786],
            g_r=[0.474],
            i_r=[-0.119],
            z_r=[-0.126],
            y_r=[-0.131],
        )

        S_type = PopulationConfig.from_kwargs(
            config_id=["default"],
            ast_class=["S"],
            albedo_min=[0.10],
            albedo_max=[0.22],
            percentage=[0.5],
            u_r=[2.182],
            g_r=[0.65],
            i_r=[-0.2],
            z_r=[-0.146],
            y_r=[-0.151],
        )

        return qv.concatenate([C_type, S_type])


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


def determine_ast_class(percent_C: float, percent_S: float, seed: int = 13612) -> str:
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
    rng = np.random.default_rng(seed)
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


def load_config(file_path: str, run_id: str = None) -> PopulationConfig:
    """
    Load the impact study configuration from a file.

    Parameters
    ----------
    file_path : str
        Path to the configuration file.
    run_id : str, optional
        User-defined run ID. If not provided, the run ID will be extracted from the file name.

    Returns
    -------
    config : `~adam_impact_study.physical_params.PopulationConfig`
        Configuration object.
    """

    if run_id is None:
        run_id = os.path.basename(file_path).split(".")[0]

    with open(file_path, "r") as file:
        config_data = json.load(file)

    S_type = PopulationConfig.from_kwargs(
        config_id=[f"S_type_{run_id}"],
        ast_class=["S"],
        albedo_min=[config_data.get("S_albedo_min", 0.10)],
        albedo_max=[config_data.get("S_albedo_max", 0.22)],
        albedo_distribution=["uniform"],
        percentage=[config_data.get("percent_S", 0.5)],
        min_diam=[config_data.get("min_diam", 0.01)],
        max_diam=[config_data.get("max_diam", 1)],
        u_r=[config_data.get("u_r_S", 2.182)],
        g_r=[config_data.get("g_r_S", 0.65)],
        i_r=[config_data.get("i_r_S", -0.2)],
        z_r=[config_data.get("z_r_S", -0.146)],
        y_r=[config_data.get("y_r_S", -0.151)],
    )

    C_type = PopulationConfig.from_kwargs(
        config_id=[f"C_type_{run_id}"],
        ast_class=["C"],
        albedo_min=[config_data.get("C_albedo_min", 0.03)],
        albedo_max=[config_data.get("C_albedo_max", 0.09)],
        albedo_distribution=["uniform"],
        percentage=[config_data.get("percent_C", 0.5)],
        min_diam=[config_data.get("min_diam", 0.01)],
        max_diam=[config_data.get("max_diam", 1)],
        u_r=[config_data.get("u_r_C", 1.786)],
        g_r=[config_data.get("g_r_C", 0.474)],
        i_r=[config_data.get("i_r_C", -0.119)],
        z_r=[config_data.get("z_r_C", -0.126)],
        y_r=[config_data.get("y_r_C", -0.131)],
    )

    config = qv.concatenate([S_type, C_type])

    return config


def generate_population(
    orbits: Orbits,
    impact_dates: Timestamp,
    population_config: PopulationConfig,
    diameters: List[float] = [0.01, 0.05, 0.14, 0.25, 0.5, 1.0],
    seed: int = 0,
    variants: int = 1,
) -> ImpactorOrbits:
    """
    Generate a population of impactors from a set of orbits and impact dates. Each orbit is duplicated once per size bin with a
    a random diameter assigned between the size bin's minimum and maximum. Optionally, generate multiple variants of each orbit with
    different physical parameters within the size bin.

    Parameters
    ----------
    orbits : Orbits
        The orbits to generate the population from.
    impact_dates : Timestamp
        The impact dates to generate the population for.
    population_config : PopulationConfig
        The population configuration to use.
    diameter_bins : List[float], optional
        The diameter bins (in km) to use for the population generation.
    seed : int, optional
        The seed to use for the population generation.
    variants : int, optional
        The number of variants to generate for each orbit.

    Returns
    -------
    ImpactorOrbits
        The generated population of impactors.
    """
    # TODO: Add support for more than just C and S-type asteroids
    S_type = population_config.select("ast_class", "S")
    C_type = population_config.select("ast_class", "C")

    if len(diameters) < 2:
        raise ValueError("At least two diameter bins must be provided")

    if len(diameters) > 999:
        raise ValueError("Too many diameter bins requested")

    if variants > 999999:
        raise ValueError("Too many variants requested")

    impactor_orbits = ImpactorOrbits.empty()
    for orbit, impact_date in zip(orbits, impact_dates):
        for i, diameter in enumerate(diameters):
            for variant in range(variants):
                # Create a unique identifier for the variant
                variant_id = f"{orbit.orbit_id[0].as_py()}_b{i:03d}_v{variant:06d}"

                # Generate a random seed based on on object id
                variant_seed = seed_from_string(variant_id, seed)

                # Determine the asteroid's taxonomic type
                ast_class = determine_ast_class(
                    C_type.percentage[0].as_py(),
                    S_type.percentage[0].as_py(),
                    variant_seed,
                )

                if ast_class == "C":
                    config = C_type
                elif ast_class == "S":
                    config = S_type

                albedo = select_albedo_from_range(
                    config.albedo_min.to_numpy()[0],
                    config.albedo_max.to_numpy()[0],
                    variant_seed,
                )

                # Determine the asteroid's absolute magnitude
                H = calculate_H(diameter, albedo)

                # Create the impactor orbits table
                impactor_orbit = ImpactorOrbits.from_kwargs(
                    orbit_id=[variant_id],
                    object_id=orbit.object_id,
                    coordinates=orbit.coordinates,
                    impact_time=impact_date,
                    dynamical_class=orbit.dynamical_class().astype("object"),
                    ast_class=[ast_class],
                    diameter=[diameter],
                    albedo=[albedo],
                    H_r=[H],
                    u_r=config.u_r,
                    g_r=config.g_r,
                    i_r=config.i_r,
                    z_r=config.z_r,
                    y_r=config.y_r,
                    GS=[0.15],
                )

                impactor_orbits = qv.concatenate([impactor_orbits, impactor_orbit])
                if impactor_orbits.fragmented():
                    impactor_orbits = qv.defragment(impactor_orbits)

    return impactor_orbits
