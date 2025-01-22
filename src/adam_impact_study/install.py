import logging
import os
import subprocess
from pathlib import Path

import pkg_resources

logger = logging.getLogger(__name__)


def build_fo():
    """Build and install Find Orb after package installation."""
    try:
        # Look for build_fo.sh in the same directory as this script
        script_path = Path(__file__).parent / "build_fo.sh"

        if not script_path.exists():
            logger.error(f"Could not find build_fo.sh at {script_path}")
            return

        # Create find_orb directory in the correct relative location
        working_dir = Path(__file__).parent.parent.parent
        os.makedirs(working_dir / "find_orb", exist_ok=True)

        subprocess.run(["bash", str(script_path)], check=True, cwd=working_dir)
        logger.info("Successfully built and installed Find Orb")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build Find Orb: {e}")
        raise
