import matplotlib.pyplot as plt
import pyarrow.compute as pc

from adam_impact_study.impacts_study import ImpactStudyResults


def plot_ip_over_time(impact_study_results: ImpactStudyResults) -> None:
    # change to x axis #days before impact
    """
    Plot the impact probability (IP) over time for each object in the provided observations.

    Parameters
    ----------
    impact_study_results : `~quiver.Table`
        Table containing the impact study results with columns 'object_id', 'day', and 'impact_probability

    Returns
    -------
    None
        This function does not return any value. It generates and displays plots for each object.
    """
    object_ids = impact_study_results.object_id.unique()

    for obj in object_ids:
        print("Object ID Plotting: ", obj)
        plt.figure()
        ips = impact_study_results.apply_mask(
            pc.equal(impact_study_results.object_id, obj)
        )
        plt.scatter(ips.day, ips.impact_probability)
        plt.title(obj)
        plt.xlabel("Day")
        plt.ylabel("Impact Probability")
        plt.plot(ips.day, ips.impact_probability)
        plt.savefig(f"IP_{obj}.png")
        plt.close()
