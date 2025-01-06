import logging
import os

import matplotlib.pyplot as plt
import pyarrow.compute as pc

from adam_impact_study.types import ImpactStudyResults
from adam_impact_study.utils import get_study_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_ip_over_time(impact_study_results: ImpactStudyResults, run_dir: str) -> None:
    """
    Plot the impact probability (IP) over time for each object in the provided observations.

    Parameters
    ----------
    impact_study_results : `~quiver.Table`
        Table containing the impact study results with columns 'object_id', 'day', and 'impact_probability
    run_dir : str
        Directory for this study run

    Returns
    -------
    None
        This function does not return any value. It generates and displays plots for each object.
    """
    object_ids = impact_study_results.object_id.unique().to_pylist()

    for object_id in object_ids:
        paths = get_study_paths(run_dir, object_id)
        object_dir = paths["object_base_dir"]
        logger.info(f"Object ID Plotting: {object_id}")
        plt.figure()

        # Get data for this object
        ips = impact_study_results.apply_mask(
            pc.equal(impact_study_results.object_id, object_id)
        )

        # Sort by observation end time
        mjd_times = ips.observation_end.mjd().to_numpy(zero_copy_only=False)
        probabilities = ips.impact_probability.to_numpy(zero_copy_only=False)
        sort_indices = mjd_times.argsort()
        mjd_times = mjd_times[sort_indices]
        probabilities = probabilities[sort_indices]

        # Plot sorted data
        plt.scatter(mjd_times, probabilities)
        plt.title(object_id)
        plt.xlabel("Day")
        plt.ylabel("Impact Probability")
        plt.plot(mjd_times, probabilities)

        plt.savefig(os.path.join(object_dir, f"IP_{object_id}.png"))
        plt.close()
