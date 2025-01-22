import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def build_fo():
    """Build and install Find Orb after package installation."""
    try:
        script_path = Path(__file__).parent.parent.parent / "build_fo.sh"
        if not script_path.exists():
            logger.error(f"Could not find build_fo.sh at {script_path}")
            return

        subprocess.run(["bash", str(script_path)], check=True)
        logger.info("Successfully built and installed Find Orb")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build Find Orb: {e}")
        raise
