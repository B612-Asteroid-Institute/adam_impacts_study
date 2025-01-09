import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional, Tuple

from adam_core.observations.ades import ADESObservations
from adam_core.orbits import Orbits

from adam_impact_study.conversions import (
    fo_to_adam_orbit_cov,
    od_observations_to_ades_file,
    rejected_observations_from_fo,
)

from .types import Observations

logger = logging.getLogger(__name__)


FO_BINARY_DIR = pathlib.Path(__file__).parent.parent.parent / "find_orb/find_orb"
LINUX_JPL_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "find_orb/.find_orb/linux_p1550p2650.430t"
)


def _populate_fo_directory(working_dir: str) -> str:
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
    ]

    # Copy required files to the current directory
    for file in required_files:
        src = os.path.join(FO_BINARY_DIR, file)
        dst = os.path.join(working_dir, file)
        shutil.copy2(src, dst)  # Copy to current directory

    # Template in the JPL path to environ.dat
    environ_dat_template = pathlib.Path(__file__).parent / "environ.dat.tpl"
    with open(environ_dat_template, "r") as file:
        environ_dat_content = file.read()
    environ_dat_content = environ_dat_content.format(
        LINUX_JPL_FILENAME=LINUX_JPL_PATH.absolute()
    )
    with open(os.path.join(working_dir, "environ.dat"), "w") as file:
        file.write(environ_dat_content)

    return working_dir


def _create_fo_tmp_directory() -> str:
    """
    Creates a temporary directory that avoids /tmp to handle fo locking and directory length limits.
    Uses ~/.cache/adam_impact_study/ftmp to avoid Find_Orb's special handling of paths containing /tmp/.

    Returns:
        str: The absolute path to the temporary directory populated with necessary FO files
    """
    base_tmp_dir = os.path.expanduser("~/.cache/adam_impact_study/ftmp")
    os.makedirs(base_tmp_dir, mode=0o770, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir)
    os.chmod(tmp_dir, 0o770)
    tmp_dir = _populate_fo_directory(tmp_dir)
    return tmp_dir


def run_fo_od(
    observations: Observations,
    paths: dict,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """Run Find_Orb orbit determination with directory-based paths

    Parameters
    ----------
    observations : Observations
        Observations to process
    paths : dict
        Dictionary containing paths for input/output files, as returned by get_study_paths()

    Returns
    -------
    Tuple[Orbits, ADESObservations, Optional[str]]
        Tuple containing:
        - Determined orbit
        - Processed observations
        - Error message (if any)
    """
    fo_tmp_dir = _create_fo_tmp_directory()

    # Create input file
    input_file = os.path.join(fo_tmp_dir, "observations.csv")
    od_observations_to_ades_file(observations, input_file)

    # Run Find_Orb
    fo_command = (
        f"{FO_BINARY_DIR}/fo {input_file} -c "
        f"-D {fo_tmp_dir}/environ.dat "
        f"-O {fo_tmp_dir}"
    )

    logger.info(f"fo command: {fo_command}")

    result = subprocess.run(
        fo_command,
        shell=True,
        cwd=fo_tmp_dir,
        text=True,
        capture_output=True,
    )
    logger.debug(f"{result.stdout}\n{result.stderr}")

    # copy all the files to our fo_dir
    shutil.copytree(fo_tmp_dir, paths["fo_dir"], dirs_exist_ok=True)
    if result.returncode != 0:
        logger.warning(f"Find_Orb failed with return code {result.returncode}")
        logger.warning(f"{result.stdout}\n{result.stderr}")
        return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"

    if not os.path.exists(f"{paths['fo_dir']}/covar.json") or not os.path.exists(
        f"{paths['fo_dir']}/total.json"
    ):
        logger.warning("Find_Orb failed, covar.json or total.json file not found")
        return (
            Orbits.empty(),
            ADESObservations.empty(),
            "Find_Orb failed, covar.json or total.json file not found",
        )

    orbit = fo_to_adam_orbit_cov(paths["fo_dir"])
    rejected = rejected_observations_from_fo(paths["fo_dir"])

    return orbit, rejected, None
