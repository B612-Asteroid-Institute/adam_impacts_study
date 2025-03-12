import hashlib
import os
import pathlib
from typing import Optional


def seed_from_string(s: str, seed: Optional[int] = None) -> int:
    """
    Generate a random seed integer from an string (typically representing an orbit or object ID)

    Parameters
    ----------
    s : str
        Object ID to generate a seed from.
    seed : int, optional
        Seed to add to the generated seed (default: 0).

    Returns
    -------
    int
        Seed integer generated from the object ID.
    """
    if seed is None:
        seed = 0
    hast_str = hashlib.sha256(s.encode())
    hash_int = int(hast_str.hexdigest(), 16)
    return (hash_int + seed) % 2**32


def get_study_paths(
    run_dir: str, orbit_id: str, time_range: Optional[str] = None
) -> dict[str, pathlib.Path]:
    """Get standardized paths for impact study results.

    Parameters
    ----------
    run_dir : str
        Directory for this specific study run
    orbit_id : str
        Orbit identifier
    time_range : str, optional
        Time range in format "mjd_start__mjd_end"

    Returns
    -------
    dict
        Dictionary containing all relevant paths
    """
    assert pathlib.Path(run_dir).is_absolute(), "run_dir must be an absolute path"
    obj_dir = pathlib.Path(run_dir) / orbit_id

    paths = {
        "orbit_base_dir": obj_dir,
        "sorcha_dir": obj_dir / "sorcha",
    }

    if time_range:
        time_dir = obj_dir / "windows" / time_range
        paths.update(
            {
                "time_dir": time_dir,
                "fo_dir": time_dir / "fo",
            }
        )

    # Create directories
    for path in paths.values():
        if not str(path).endswith(".parquet"):
            path.mkdir(parents=True, exist_ok=True)

    return paths
