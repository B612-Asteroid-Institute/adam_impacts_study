import logging
import os
import pathlib
import shutil
import subprocess
from typing import Optional, Tuple

from adam_core.observations.ades import ADESObservations
from adam_core.orbits import Orbits

from adam_impact_study.conversions import (
    fo_to_adam_orbit_cov,
    od_observations_to_ades_file,
    rejected_observations_from_fo,
)

from .types import Observations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LINUX_JPL_PATH = pathlib.Path(__file__).parent.parent.parent / "find_orb/linux_p1550p2650.430t"


def _create_fo_working_directory(FO_DIR: str, working_dir: str) -> str:
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
        src = os.path.join(FO_DIR, file)
        dst = os.path.join(working_dir, file)
        shutil.copy2(src, dst)  # Copy to current directory

    # Template in the JPL path to environ.dat
    environ_dat_template = pathlib.Path(__file__).parent / "environ.dat.tpl"
    with open(environ_dat_template, "r") as file:
        environ_dat_content = file.read()
    environ_dat_content = environ_dat_content.format(LINUX_JPL_FILENAME=LINUX_JPL_PATH.absolute())
    with open(os.path.join(working_dir, "environ.dat"), "w") as file:
        file.write(environ_dat_content)

    return working_dir


def run_fo_od(
    observations: Observations,
    fo_dir: str,
    paths: dict,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """Run Find_Orb orbit determination with directory-based paths"""
    
    # Create input file
    input_file = os.path.join(paths['fo_inputs'], "observations.csv")
    od_observations_to_ades_file(observations, input_file)
    
    # Run Find_Orb
    fo_command = (
        f"{fo_dir}/fo {input_file} -c "
        f"-D {paths['fo_outputs']}/environ.dat"
    )
    
    result = subprocess.run(
        fo_command,
        shell=True,
        cwd=paths['fo_outputs'],
        text=True,
        capture_output=True,
    )
    logger.debug(f"{result.stdout}\n{result.stderr}")
    if result.returncode != 0 or not os.path.exists(f"{paths['fo_outputs']}/covar.json"):
        logger.warning(f"{result.stdout}\n{result.stderr}")
        return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"
        
    orbit = fo_to_adam_orbit_cov(paths['fo_outputs'])
    rejected = rejected_observations_from_fo(paths['fo_outputs'])
    
    return orbit, rejected, None
