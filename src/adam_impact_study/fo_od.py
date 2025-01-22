import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import pyarrow.compute as pc
from adam_core.observations.ades import ADESObservations
from adam_core.orbits import Orbits

from adam_impact_study.conversions import (
    fo_to_adam_orbit_cov,
    od_observations_to_ades_file,
    rejected_observations_from_fo,
)

from .types import Observations

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))


FO_BINARY_DIR = pathlib.Path(__file__).parent.parent.parent / "find_orb/find_orb"
LINUX_JPL_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "find_orb/.find_orb/linux_p1550p2650.440t"
)

BC405_FILENAME = (
    pathlib.Path(__file__).parent.parent.parent / "find_orb/.find_orb/bc405.dat"
)


def _populate_fo_directory(working_dir: str, fo_binary_dir: str = FO_BINARY_DIR, bc405_filename: str = BC405_FILENAME, linus_jpl_path: str = LINUX_JPL_PATH) -> str:
    os.makedirs(working_dir, exist_ok=True)
    # List of required files to copy from FO_DIR to current directory
    required_files = [
        "ObsCodes.htm",
        "jpl_eph.txt",
        "orbitdef.sof",
        "rovers.txt",
        "xdesig.txt",
        "cospar.txt",
        "efindorb.txt",
        "odd_name.txt",
        "sigma.txt",
        "mu1.txt",
    ]

    # Copy required files to the current directory
    for file in required_files:
        src = os.path.join(fo_binary_dir, file)
        dst = os.path.join(working_dir, file)
        shutil.copy2(src, dst)  # Copy to current directory

    # Copy bc405.dat to the current directory
    shutil.copy2(bc405_filename, os.path.join(working_dir, "bc405.dat"))

    # Template in the JPL path to environ.dat
    environ_dat_template = pathlib.Path(__file__).parent / "environ.dat.tpl"
    with open(environ_dat_template, "r") as file:
        environ_dat_content = file.read()
    environ_dat_content = environ_dat_content.format(
        LINUX_JPL_FILENAME=linus_jpl_path.absolute(),
    )
    with open(os.path.join(working_dir, "environ.dat"), "w") as file:
        file.write(environ_dat_content)

    return working_dir


def _create_fo_tmp_directory(fo_binary_dir: str = FO_BINARY_DIR, bc405_filename: str = BC405_FILENAME, linus_jpl_path: str = LINUX_JPL_PATH) -> str:
    """
    Creates a temporary directory that avoids /tmp to handle fo locking and directory length limits.
    Uses ~/.cache/adam_impact_study/ftmp to avoid Find_Orb's special handling of paths containing /tmp/.

    Returns:
        str: The absolute path to the temporary directory populated with necessary FO files
    """
    base_tmp_dir = os.path.expanduser("~/.cache/adam_impact_study/")
    os.makedirs(base_tmp_dir, mode=0o770, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir, prefix="fo_")
    os.chmod(tmp_dir, 0o770)
    tmp_dir = _populate_fo_directory(tmp_dir, fo_binary_dir, bc405_filename, linus_jpl_path)
    return tmp_dir


def _copy_files_from_tmp_to_fo_dir(fo_tmp_dir: str, fo_dir: str):
    # Selectively copy files from fo_tmp_dir to fo_dir
    # Explicitly remove the tmp directory after copying
    files_to_copy = [
        "covar.txt",
        "cospar.txt",
        "elem_short.json",
        "gauss.out",
        "covar.json",
        "total.json",
        "elements.txt",
        "debug.txt",
        "bc405pre.dat",
        "environ.dat",
        # Copy observations for fo debugging purposes
        "observations.ades",
    ]

    for file in files_to_copy:
        if os.path.exists(os.path.join(fo_tmp_dir, file)):
            shutil.copy2(os.path.join(fo_tmp_dir, file), os.path.join(fo_dir, file))


def _de440t_exists():
    if not os.path.exists(LINUX_JPL_PATH):
        raise Exception(
            f"DE440t file not found at {LINUX_JPL_PATH}, find_orb will not work correctly"
        )


def run_fo_od(
    observations: Observations,
    fo_result_dir: str,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """Run Find_Orb orbit determination with directory-based paths

    Parameters
    ----------
    observations : Observations
        Observations to process
    fo_result_dir : str
        Directory where Find_Orb output files will be written

    Returns
    -------
    Tuple[Orbits, ADESObservations, Optional[str]]
        Tuple containing:
        - Determined orbit
        - Processed observations
        - Error message (if any)
    """

    # This function is only valid for a single orbit_id
    if len(observations.orbit_id.unique()) > 1:
        raise ValueError("This function is only valid for a single orbit_id")

    _de440t_exists()
    fo_tmp_dir = _create_fo_tmp_directory()

    # Create input file
    input_file = os.path.join(fo_tmp_dir, "observations.ades")
    # Truncate object_id to 8 characters. we will re-assign it after FO runs
    orbit_id = observations.orbit_id
    observations = observations.set_column(
        "orbit_id", pc.utf8_slice_codeunits(orbit_id, 0, 8)
    )
    od_observations_to_ades_file(observations, input_file)

    debug_level = os.environ.get("ADAM_LOG_LEVEL", "INFO")
    fo_debug_level = {
        "DEBUG": 10,
        "INFO": 2,
    }.get(debug_level, 0)
    # Run Find_Orb
    fo_command = (
        f"{FO_BINARY_DIR}/fo {input_file} -c "
        f"-d {fo_debug_level} "
        f"-D {fo_tmp_dir}/environ.dat "
        f"-O {fo_tmp_dir}"
    )

    min_mjd = observations.coordinates.time.min().mjd()[0].as_py()
    max_mjd = observations.coordinates.time.max().mjd()[0].as_py()
    logger.info(
        f"Running fo for {orbit_id[0].as_py()} from {min_mjd} to {max_mjd}"
    )
    logger.debug(f"fo command: {fo_command}")

    result = subprocess.run(
        fo_command,
        shell=True,
        cwd=fo_tmp_dir,
        text=True,
        capture_output=True,
    )
    logger.debug(f"{result.stdout}\n{result.stderr}")

    _copy_files_from_tmp_to_fo_dir(fo_tmp_dir, fo_result_dir)
    # Remove the tmp directory after copying because it has
    # some large files in it that we don't need
    shutil.rmtree(fo_tmp_dir)

    if result.returncode != 0:
        logger.warning(
            f"Find_Orb failed with return code {result.returncode} for {len(observations)} observations in {fo_result_dir}"
        )
        logger.warning(f"{result.stdout}\n{result.stderr}")
        return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"

    if not os.path.exists(f"{fo_result_dir}/covar.json") or not os.path.exists(
        f"{fo_result_dir}/total.json"
    ):
        logger.warning("Find_Orb failed, covar.json or total.json file not found")
        return (
            Orbits.empty(),
            ADESObservations.empty(),
            "Find_Orb failed, covar.json or total.json file not found",
        )

    orbit = fo_to_adam_orbit_cov(fo_result_dir)

    # Re-assign orbit_id to the original value
    orbit = orbit.set_column("orbit_id", observations[0].orbit_id)
    rejected = rejected_observations_from_fo(fo_result_dir)

    return orbit, rejected, None
