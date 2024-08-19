import matplotlib.pyplot as plt
import pandas as pd


def plot_ip_over_time(ip_dict):
    #change to x axis #days before impact
    """
    Plot the impact probability (IP) over time for each object in the provided dictionary.

    Parameters
    ----------
    ip_dict : dict
        Dictionary where keys are object IDs and values are dictionaries with
        days as keys and impact probabilities (IP) as values.

    Returns
    -------
    None
        This function does not return any value. It generates and displays plots for each object.
    """
    for obj in ip_dict.keys():
        obj_dict = ip_dict[obj]
        ip_df = pd.DataFrame(list(obj_dict.items()), columns=["Day", "IP"])
        plt.scatter(ip_df["Day"], ip_df["IP"])
        plt.title(obj)
        plt.xlabel("Day")
        plt.ylabel("Impact Probability")
        plt.plot(ip_df["Day"], ip_df["IP"])
        plt.savefig(f"IP_{obj}.png")
