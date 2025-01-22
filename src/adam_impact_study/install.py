import logging
import os
import subprocess
from pathlib import Path

import pkg_resources

logger = logging.getLogger(__name__)


def build_fo():
    """Build and install Find Orb after package installation."""
    try:
        # First try to find build_fo.sh in package resources
        try:
            script_path = pkg_resources.resource_filename(
                "adam_impact_study", "../build_fo.sh"
            )
        except pkg_resources.DistributionNotFound:
            # Fall back to development path
            script_path = str(Path(__file__).parent.parent.parent / "build_fo.sh")

        if not os.path.exists(script_path):
            logger.error(f"Could not find build_fo.sh at {script_path}")
            return

        subprocess.run(["bash", script_path], check=True)
        logger.info("Successfully built and installed Find Orb")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build Find Orb: {e}")
        raise
