import logging
import os

import matplotlib.pyplot as plt
import pyarrow.compute as pc

from adam_impact_study.types import ImpactStudyResults
from adam_impact_study.utils import get_study_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_ip_over_time(impact_study_results: ImpactStudyResults, base_dir: str, run_name: str) -> None:
    # change to x axis #days before impact
    """
    Plot the impact probability (IP) over time for each object in the provided observations.

    Parameters
    ----------
    impact_study_results : `~quiver.Table`
        Table containing the impact study results with columns 'object_id', 'day', and 'impact_probability
    base_dir : str
        Base directory for all results
    run_name : str
        Name of the study run

    Returns
    -------
    None
        This function does not return any value. It generates and displays plots for each object.
    """
    object_ids = impact_study_results.object_id.unique().to_pylist()

    for object_id in object_ids:
        paths = get_study_paths(base_dir, run_name, object_id)
        object_dir = paths["object_base_dir"]
        logger.info(f"Object ID Plotting: {object_id}")
        plt.figure()
        ips = impact_study_results.apply_mask(
            pc.equal(impact_study_results.object_id, object_id)
        )
        plt.scatter(
            ips.observation_end.mjd().to_numpy(zero_copy_only=False),
            ips.impact_probability.to_numpy(zero_copy_only=False),
        )
        plt.title(object_id)
        plt.xlabel("Day")
        plt.ylabel("Impact Probability")
        plt.plot(
            ips.observation_end.mjd().to_numpy(zero_copy_only=False),
            ips.impact_probability.to_numpy(zero_copy_only=False),
        )
        plt.savefig(os.path.join(object_dir, f"IP_{object_id}.png"))
        plt.close()
