import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.constants import DATA_PATH

#create function to plot mean loss curve with std as confidence interval (shaded)
def plot_loss_curve(gs_loss, title):
    gs_loss_mean = np.mean(gs_loss, axis=0)
    gs_loss_std = np.std(gs_loss, axis=0)
    gs_loss_df = pd.DataFrame({"loss": gs_loss_mean, "std": gs_loss_std})
    gs_loss_df["iter"] = gs_loss_df.index
    sns.lineplot(data=gs_loss_df, x="iter", y="loss")
    plt.fill_between(gs_loss_df["iter"], gs_loss_df["loss"] - gs_loss_df["std"], gs_loss_df["loss"] + gs_loss_df["std"], alpha=0.5)
    plt.title(title)
    plt.show()

#plot loss curves for different gs values
gs_10000_loss = np.load(DATA_PATH / "gs_ex2/gs_10000/cond_loss10000.npy")
plot_loss_curve(gs_10000_loss, "gs=10000")

gs_1000_loss = np.load(DATA_PATH / "gs_ex2/gs_1000/cond_loss1000.npy")
plot_loss_curve(gs_1000_loss, "gs=1000")   

gs_100_loss = np.load(DATA_PATH / "gs_ex2/gs_100/cond_loss100.npy")
plot_loss_curve(gs_100_loss, "gs=100")

gs_10_loss = np.load(DATA_PATH / "gs_ex2/gs_10/cond_loss10.npy")
plot_loss_curve(gs_10_loss, "gs=10")