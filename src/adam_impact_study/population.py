import os
import json
import logging
from typing import List

import numpy as np
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
import numpy.typing as npt
import ray
import multiprocessing as mp
from typing import Optional, Union
from adam_core.utils.iter import _iterate_chunks
from adam_core.ray_cluster import initialize_use_ray
from adam_impact_study.types import ImpactorOrbits
from .utils import seed_from_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopulationConfig(qv.Table):
    config_id = qv.LargeStringColumn()
    ast_class = qv.LargeStringColumn()
    albedo_min = qv.Float64Column(nullable=True)
    albedo_max = qv.Float64Column(nullable=True)
    albedo_scale_factor = qv.Float64Column(nullable=True)
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
            albedo_scale_factor=[0.029],  # d in Wright's paper
            percentage=[0.233],
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
            albedo_scale_factor=[0.170],  # b in Wright's paper
            percentage=[0.767],
            u_r=[2.182],
            g_r=[0.65],
            i_r=[-0.2],
            z_r=[-0.146],
            y_r=[-0.151],
        )

        return qv.concatenate([C_type, S_type])


def select_albedo_from_range(
    albedo_min: float, albedo_max: float, rng: np.random.Generator = None
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
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(albedo_min, albedo_max)

def select_albedo_rayleigh(scale: float, rng: np.random.Generator = None):
    """
    Sample albedo using a Rayleigh distribution.
    Parameters
    ----------
    scale : float
        The scale parameter for the Rayleigh distribution.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    albedo : float
        The sampled albedo value.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.rayleigh(scale=scale)


def determine_ast_class(percent_C: float, percent_S: float, rng: np.random.Generator = None) -> str:
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
    # Note: when using the bimodal distribution described in Wright et al:
    # p(pV) = fd*(pV/d²)*exp(-pV²/2d²) + (1-fd)*(pV/b²)*exp(-pV²/2b²)
    # percent_C is equivalent to fd (the fraction of dark asteroids) and
    # percent_S is equivalent to 1 - fd.
    assert percent_C + percent_S == 1, "Percentage of C and S asteroids must equal 1"
    if rng is None:
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
        percentage=[config_data.get("percent_S", 0.767)],
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
        percentage=[config_data.get("percent_C", 0.233)],
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


def population_worker(
    orbit_indices: npt.NDArray[np.int64],
    orbits: Orbits,
    impact_dates: Timestamp,
    population_config: PopulationConfig,
    diameters: List[float],
    seed: int,
    variants: int,
    albedo_distribution: str,
) -> ImpactorOrbits:
    """Worker function to generate population for a chunk of orbits. 
    Each orbit is duplicated once per size bin with a random diameter 
    assigned between the size bin's minimum and maximum. Optionally, 
    generate multiple variants of each orbit with different physical 
    parameters within the size bin.
    
    Parameters
    ----------
    orbit_indices : npt.NDArray[np.int64]
        Indices of the orbits to process.
    orbits : Orbits
        Orbits to process.
    impact_dates : Timestamp
        Impact dates to process.
    population_config : PopulationConfig
        Population configuration.
    diameters : List[float]
        Diameters to process.
    seed : int
        Seed for the random number generator.
    variants : int
        Number of variants to process.
    albedo_distribution : str
        Albedo distribution to use.

    Returns
    -------
    impactor_orbits : ImpactorOrbits
        Impactor orbits.
    """
    orbits_chunk = orbits.take(orbit_indices)
    impact_dates_chunk = impact_dates.take(orbit_indices)
    
    impactor_orbits = ImpactorOrbits.empty()
    for orbit, impact_date in zip(orbits_chunk, impact_dates_chunk):
        for i, diameter in enumerate(diameters):
            for variant in range(variants):
                # Create a unique identifier for the variant
                variant_id = f"{orbit.orbit_id[0].as_py()}_b{i:03d}_v{variant:06d}"

                # Generate a random seed based on object id
                variant_seed = seed_from_string(variant_id, seed)

                # Initialize the random number generator from the variant seed
                rng = np.random.default_rng(variant_seed)

                # Determine the asteroid's taxonomic type
                ast_class = determine_ast_class(
                    population_config.select("ast_class", "C").percentage[0].as_py(),
                    population_config.select("ast_class", "S").percentage[0].as_py(),
                    rng,
                )

                if ast_class == "C":
                    config = population_config.select("ast_class", "C")
                elif ast_class == "S":
                    config = population_config.select("ast_class", "S")

                if albedo_distribution == "rayleigh":
                    albedo = select_albedo_rayleigh(
                        config.albedo_scale_factor.to_numpy()[0],
                        rng,
                    )
                elif albedo_distribution == "uniform":
                    albedo = select_albedo_from_range(
                        config.albedo_min.to_numpy()[0],
                        config.albedo_max.to_numpy()[0],
                        rng,
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

# Create remote version of worker
population_worker_remote = ray.remote(population_worker)

def generate_population(
    orbits: Union[Orbits, ray.ObjectRef],
    impact_dates: Union[Timestamp, ray.ObjectRef],
    population_config: PopulationConfig,
    diameters: List[float] = [0.01, 0.05, 0.14, 0.25, 0.5, 1.0],
    seed: int = 0,
    variants: int = 1,
    albedo_distribution: str = "rayleigh",
    chunk_size: int = 100,
    max_processes: Optional[int] = 1,
) -> ImpactorOrbits:
    """
    Generate a population of impactors from a set of orbits and impact dates.
    
    Parameters
    ----------
    [previous parameters...]
    chunk_size : int, optional
        Number of orbits to process in each chunk.
    max_processes : int, optional
        Maximum number of processes to use. If None, uses all available CPUs.
    """
    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(num_cpus=max_processes)
    
    if not use_ray:
        return population_worker(
            np.arange(len(orbits)),
            orbits,
            impact_dates,
            population_config,
            diameters,
            seed,
            variants,
            albedo_distribution,
        )

    if isinstance(orbits, ray.ObjectRef):
        orbits_ref = orbits
        orbits = ray.get(orbits)
    else:
        orbits_ref = ray.put(orbits)

    if isinstance(impact_dates, ray.ObjectRef):
        impact_dates_ref = impact_dates
        impact_dates = ray.get(impact_dates)
    else:
        impact_dates_ref = ray.put(impact_dates)

    population_config_ref = ray.put(population_config)
    
    futures = []
    idx = np.arange(len(orbits))
    for idx_chunk in _iterate_chunks(idx, chunk_size):
        futures.append(
            population_worker_remote.remote(
                idx_chunk,
                orbits_ref,
                impact_dates_ref,
                population_config_ref,
                diameters,
                seed,
                variants,
                albedo_distribution,
            )
        )

    impactor_orbits = ImpactorOrbits.empty()
    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        impactor_orbits = qv.concatenate([impactor_orbits, result])
        if impactor_orbits.fragmented():
            impactor_orbits = qv.defragment(impactor_orbits)

    return impactor_orbits

