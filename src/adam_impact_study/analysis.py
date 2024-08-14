import matplotlib.pyplot as plt


def plot_ip_over_time(ip_dict):
    for obj in ip_dict.keys():
        ip_dict = ip_dict[obj]
        # for day in ip_dict.keys():
        #    ip_dict_new[day.as_py()] = ip_dict[day].cumulative_probability[0].as_py()
        print(ip_dict)
        ip_df = pd.DataFrame(list(ip_dict.items()), columns=["Day", "IP"])
        plt.scatter(ip_df["Day"], ip_df["IP"])
        plt.title(obj)
        plt.xlabel("Day")
        plt.ylabel("Impact Probability")
        plt.plot(ip_df["Day"], ip_df["IP"])
        plt.show()
