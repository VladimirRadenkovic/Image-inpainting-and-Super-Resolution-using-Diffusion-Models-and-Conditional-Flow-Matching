import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.plotstyle import matplotlib_defaults, get_dim, TEXTWIDTHS
import pickle
from loguru import logger
import argparse
from scipy.stats import ks_2samp
import plotly.express as px

palette = sns.color_palette("Dark2")
matplotlib_defaults(autoupdate=True)

def process_text(target="col"):
    targets = {"col": ("col", TEXTWIDTHS.ICML_WCB_COLUMNWIDTH),
               "text": ("text", TEXTWIDTHS.ICML_WCB_TEXTWIDTH),
               "both": (("col", TEXTWIDTHS.ICML_WCB_COLUMNWIDTH), ("text", TEXTWIDTHS.ICML_WCB_TEXTWIDTH))}

    target_data = targets.get(target)

    if not target_data:
        raise ValueError(f"Invalid target: {target}. Choose from 'col', 'text', or 'both'")

    if target != "both":
        target_data = [target_data]  # Single target, but we want it in a list format for consistency
    return target_data

def plot_mean_chain_distances(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):

    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))
        xmin, xmax = 3.5, 4.0

        sns.histplot(sample_df.query(f"(ca_distance_mean > {xmin}) & (ca_distance_mean < {xmax})").ca_distance_mean, 
                     ax=ax, stat="density", color=palette[0], binwidth=0.005, label="Uncond.", alpha=0.8)
        sns.histplot(ref_df.query(f"(ca_distance_mean > {xmin}) & (ca_distance_mean < {xmax})").ca_distance_mean, 
                     ax=ax, stat="density", color=palette[2], binwidth=0.005, label="CATH", alpha=0.8)
        if cond_df:
            sns.histplot(cond_df.query(f"(ca_distance_mean > {xmin}) & (ca_distance_mean < {xmax})").ca_distance_mean, 
                        ax=ax, stat="density", color=palette[1], binwidth=0.005, label="Cond.", alpha=0.8)

        plt.xlabel(r"Backbone mean $C_\alpha$-distance [\AA]")
        plt.legend(loc="upper left", frameon=False)
        plt.xlim(xmin, xmax)
        sns.despine()
        plt.savefig(f"{plot_dir}/backbone_dist_mean_{name}.pdf", bbox_inches="tight")

def plot_mean_ca_angles(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))
        xmin, xmax = 50,100

        sns.histplot(sample_df.query(f"(ca_angle_mean > {xmin}) & (ca_angle_mean < {xmax})").ca_angle_mean, 
                     ax=ax, stat="density", color=palette[0], binwidth=2, label="Uncond.", alpha=0.8)
        sns.histplot(ref_df.query(f"(ca_angle_mean > {xmin}) & (ca_angle_mean < {xmax})").ca_angle_mean, 
                     ax=ax, stat="density", color=palette[2], binwidth=2, label="CATH", alpha=0.8)
        if cond_df:
            sns.histplot(cond_df.query(f"(ca_angle_mean > {xmin}) & (ca_angle_mean < {xmax})").ca_angle_mean, 
                        ax=ax, stat="density", color=palette[1], binwidth=2, label="Cond.", alpha=0.8)

        plt.xlabel(r"Backbone mean $C_\alpha$-angle [degrees]")
        plt.legend(loc="upper right", frameon=False)
        plt.xlim(xmin, xmax)
        sns.despine()
        plt.savefig(f"{plot_dir}/backbone_angle_mean_{name}.pdf", bbox_inches="tight")

def plot_secondary_structure_usage(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))

        sse_orig = ref_df[["helix_proportion", "sheet_proportion", "coil_proportion"]].mean().values
        sse_samples = sample_df[["helix_proportion", "sheet_proportion", "coil_proportion"]].mean().values
        if cond_df:
            sse_cond = cond_df[["helix_proportion", "sheet_proportion", "coil_proportion"]].mean().values
            sse_usage = np.vstack([sse_samples, sse_cond, sse_orig])
        else:
            sse_usage = np.vstack([sse_samples, sse_orig])


        # Add percentage labels
        # Calculate middle y position of each bar
        y_pos = sse_usage.cumsum(axis=1) - sse_usage/2
        if cond_df:
            plt.bar(np.arange(3), sse_usage[:,0], label="Helix", color=(1.0,0.6,0.6), width=0.6)
            plt.bar(np.arange(3), sse_usage[:,1], bottom=sse_usage[:,0], label="Sheet", color=(0.75,	0.75,	1.0), width=0.6)
            plt.bar(np.arange(3), sse_usage[:,2], bottom=sse_usage[:,0] + sse_usage[:,1], label="Coil", color=(0.8, 0.8, 0.8), width=0.6)
            plt.xticks([0,1,2], ["Unconditioned", "Conditioned", "CATH"])
            for i in range(3):
                plt.text(0, y_pos[0,i], f"{sse_usage[0,i]*100:.1f}\%", color="black", ha="center", va="center")
                plt.text(1, y_pos[1,i], f"{sse_usage[1,i]*100:.1f}\%", color="black", ha="center", va="center")
                plt.text(2, y_pos[2,i], f"{sse_usage[2,i]*100:.1f}\%", color="black", ha="center", va="center")
            plt.xticks([0,1,2], ["Unconditioned", "Conditioned", "CATH"])
            plt.xlim(-0.5,2.5)
            plt.axvline(1.5, -0.12, 1.01, color="black", linestyle="-", linewidth=0.5, clip_on=False)
        else:
            plt.bar(np.arange(2), sse_usage[:,0], label="Helix", color=(1.0,0.6,0.6), width=0.6)
            plt.bar(np.arange(2), sse_usage[:,1], bottom=sse_usage[:,0], label="Sheet", color=(0.75,	0.75,	1.0), width=0.6)
            plt.bar(np.arange(2), sse_usage[:,2], bottom=sse_usage[:,0] + sse_usage[:,1], label="Coil", color=(0.8, 0.8, 0.8), width=0.6)
            plt.xticks([0,1], ["Unconditioned", "CATH"])
            plt.xlim(-0.5,1.5)
                    
            for i in range(3):
                plt.text(0, y_pos[0,i], f"{sse_usage[0,i]*100:.1f}\%", color="black", ha="center", va="center")
                plt.text(1, y_pos[1,i], f"{sse_usage[1,i]*100:.1f}\%", color="black", ha="center", va="center")

        plt.legend()
        plt.ylim(0,1.01)
        plt.yticks([])
        #plt.legend(bbox_to_anchor=(1., 1), loc='upper left', borderaxespad=0.)
        # Make legend horizontal and put on top
        plt.legend(bbox_to_anchor=(0., -0.35, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0., frameon=True)
        sns.despine(left=True)
        plt.savefig(f"{plot_dir}/secondary_structure_usage_{name}.pdf", bbox_inches="tight")

def plot_radius_of_gyration(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))

        sns.histplot(sample_df.radius_of_gyration, ax=ax, stat="density", bins=50, color=palette[0], alpha=0.8, label="Uncond.")
        sns.histplot(ref_df.radius_of_gyration, ax=ax, stat="density", bins=50, color=palette[2], alpha=0.8, label="CATH")
        if cond_df:
            sns.histplot(cond_df.radius_of_gyration, ax=ax, stat="density", bins=50, color=palette[1], alpha=0.8, label="Cond.")

        #plt.xlim(0, 1.0)
        sns.despine(left=False)
        plt.xlabel(r"Radius of gyration $R_g$")
        plt.legend(loc="upper right", frameon=False)
        plt.savefig(f"{plot_dir}/radius_of_gyration_{name}.pdf", bbox_inches="tight")

def plot_sphericity(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))

        sns.histplot(sample_df.shpericality, ax=ax, stat="density", bins=50, color=palette[0], alpha=0.8, label="Uncond.")
        sns.histplot(ref_df.shpericality, ax=ax, stat="density", bins=50, color=palette[2], alpha=0.8, label="CATH")
        if cond_df:
            sns.histplot(cond_df.shpericality, ax=ax, stat="density", bins=50, color=palette[1], alpha=0.8, label="Cond.")

        #plt.xlim(0, 1.0)
        sns.despine(left=False)
        plt.xlabel(r"Sphericity $R_g^2 / R_{max}^2$")
        plt.legend(loc="upper left", frameon=False)
        plt.savefig(f"{plot_dir}/sphericity_{name}.pdf", bbox_inches="tight")


def plot_conditional_loss(sample_df, cond_df, loss, plot_dir=None, plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))

        sns.histplot(sample_df.loss, ax=ax, stat="density", binwidth=0.02, color=palette[0], alpha=0.8, label="Uncond.")
        sns.histplot(cond_df.loss, ax=ax, stat="density", binwidth=0.02, color=palette[1], alpha=0.8, label="Cond.")

        plt.xlim(0, 1.15)
        sns.despine(left=False)
        if loss == "nma":
            plt.xlabel(r"NMA Loss $l(y, v(x_t))$")
        elif loss == "motif":
            plt.xlabel(r"Motif Loss $l(y, v(x_t))$")
        plt.legend(loc="upper right", frameon=False)
        plt.savefig(f"{plot_dir}/nma_loss_{name}.pdf", bbox_inches="tight")

def plot_conditional_loss_vs_step(cond_df, data_dir, plot_dir=None, plot_width="col"):
    df = pd.DataFrame()
    for i in range(len(cond_df)):
        cond_loss = np.load(f"{data_dir}/cond_loss_samples/condloss_{i}.npy")
        cond_loss = cond_loss*(15**2)
        # cond_loss = np.sqrt(cond_loss)
        cond_loss = pd.DataFrame(cond_loss, columns=["cond_loss"])
        cond_loss["step"] = cond_loss.index+1
        df = pd.concat([df, cond_loss])

    target_data = process_text(target=plot_width)
    for name, width in target_data:
        #MSE plot
        fig, ax = plt.subplots(figsize=get_dim(width=width))
        sns.lineplot(x='step',y='cond_loss', ax=ax, data=df)
        plt.ylim(0, 8)
        plt.axhline(1, color="red", linestyle="--", linewidth=1,
                    xmin=0, xmax=125)
        plt.xlabel(r"Sampling step $t$")
        plt.ylabel(r"Motif MSE [$\AA^2$]")
        sns.despine(left=False)
        plt.savefig(f"{plot_dir}/cond_loss_mse_{name}.pdf", bbox_inches="tight")
        #RMSD plot
        fig, ax = plt.subplots(figsize=get_dim(width=width))
        df["cond_loss"] = np.sqrt(df["cond_loss"])
        sns.lineplot(x='step',y='cond_loss', ax=ax, data=df)
        plt.ylim(0, 3)
        plt.axhline(1, color="red", linestyle="--", linewidth=1,
                    xmin=0, xmax=125)
        plt.xlabel(r"Sampling step $t$")
        plt.ylabel(r"Motif RMSD [$\AA$]")
        sns.despine(left=False)
        plt.savefig(f"{plot_dir}/cond_loss_rmsd_{name}.pdf", bbox_inches="tight")

def plot_novelty(sample_df, cond_df, novelty_metric, plot_dir=None,plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))

        sns.histplot(sample_df[novelty_metric], ax=ax, stat="density", bins=50, color=palette[0], alpha=0.8, label="Uncond.")
        if cond_df:
            sns.histplot(sample_df[novelty_metric], ax=ax, stat="density", bins=50, color=palette[1], alpha=0.8, label="Cond.")

        #plt.xlim(0, 1.0)
        sns.despine(left=False)
        plt.xlabel(r"TM Score vs closest training structure")
        plt.legend(loc="upper right", frameon=False)
        plt.savefig(f"{plot_dir}/{novelty_metric}_{name}.pdf", bbox_inches="tight")

def calculate_similarity_metric(sample_df, ref_df, column):
    ks_stat, _ = ks_2samp(sample_df[column], ref_df[column])
    similarity_metric = 1 - ks_stat
    return similarity_metric

def calculate_mse_metric(sample_df, ref_df, column):
    if column == "exceeds_canvas":
        mse_series = sample_df[column]
    else:
        mse_series = (sample_df[column] - ref_df[column].mean()) ** 2
    return mse_series

def plot_radar(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):
    target_data = process_text(target=plot_width)
    for name, width in target_data:
        labels = ["Angles", "Distances", "SS Ratio", "Issues", "Sphericity"]
        categories = ["ca_angle_mean", "ca_distance_mean", "helix_proportion", "exceeds_canvas", "shpericality"]
        values = [calculate_similarity_metric(sample_df, ref_df, cat) for cat in categories]
        #calculate other ss ratios
        sheets = calculate_similarity_metric(sample_df, ref_df, "sheet_proportion")
        coils = calculate_similarity_metric(sample_df, ref_df, "coil_proportion")
        values[2] = (values[2] + sheets + coils) / 3
        values += values[:1]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], labels, color='grey', size=7) #Draw one axe per variable + add labels
        ax.set_rlabel_position(0) #Draw ylabels
        plt.yticks([0.25,0.5,0.75], ["1/4","1/2","3/4"], color="grey", size=6)
        plt.ylim(0,1)
        ax.plot(angles, values, linewidth=1, linestyle='solid') #plot data
        ax.fill(angles, values, 'b', alpha=0.1) #Fill area
        logger.info(f"Radar plot values: {values}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/radar_{name}.pdf", bbox_inches="tight")

def plot_parallel_coordinates(sample_df, ref_df, cond_df=None, plot_dir=None, plot_width="col"):
    #calculate MSE of average metric in sample versus average metric in ref
    labels = ["Angles", "Distances", "SS Ratio", "Issues", "Sphericity"]
    categories = ["ca_angle_mean", "ca_distance_mean", "helix_proportion", "exceeds_canvas", "shpericality"]
    values = pd.DataFrame([calculate_mse_metric(sample_df, ref_df, cat) for cat in categories])
    #calculate other ss ratios
    sheets = calculate_mse_metric(sample_df, ref_df, "sheet_proportion")
    coils = calculate_mse_metric(sample_df, ref_df, "coil_proportion")
    values[2] = (values[2] + sheets + coils) / 3
    #make dataframe with each value a new column and the labels as the column names
    df = pd.DataFrame([values], columns=labels)

    target_data = process_text(target=plot_width)
    for name, width in target_data:
        fig, ax = plt.subplots(figsize=get_dim(width=width))
        # Parallel coordinates plot
        # parallel_coordinates(df, ax=ax, color=palette[0], linewidth=3)
        # sns.despine(left=False)
        fig = px.parallel_coordinates(df, color=labels[0], dimensions=labels,
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
        sns.despine(left=False)
        fig.write_image(f"{plot_dir}/parallel_coordinates_{name}.pdf")
        # plt.savefig(f"{plot_dir}/parallel_coordinates_{name}.pdf", bbox_inches="tight")



def run_plot_pipeline(sample_df, ref_df, cond_df=None, plot_dir=None, data_dir=None, plot_width="col"):
    plot_mean_chain_distances(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    plot_secondary_structure_usage(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    plot_mean_ca_angles(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    plot_radius_of_gyration(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    plot_sphericity(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    # plot_novelty(sample_df, cond_df, "tm_score", plot_dir=plot_dir, plot_width=plot_width)
    # plot_novelty(sample_df, cond_df, "rmsd", plot_dir=plot_dir, plot_width=plot_width)
    # plot_novelty(sample_df, cond_df, "gdt_score", plot_dir=plot_dir, plot_width=plot_width)
    plot_radar(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    # plot_parallel_coordinates(sample_df, ref_df, cond_df, plot_dir=plot_dir, plot_width=plot_width)
    if data_dir:
        plot_conditional_loss_vs_step(sample_df, data_dir=data_dir, 
                                  plot_dir=plot_dir, plot_width=plot_width)
    logger.info(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_df", type=str, default="/home/ked48/rds/hpc-work/protein-diffusion/data/uncond_test_old_denoiser_no_noise_last5steps/sample_stats.csv")
    parser.add_argument("--ref_df", type=str, default="/home/ked48/rds/hpc-work/protein-diffusion/data/raw/pdb_domain_ca_coords_v2023-04-25_stats.csv")
    parser.add_argument("--cond_df", type=str, required=False, default=None)
    parser.add_argument("--plot_dir", type=str, default="/home/ked48/rds/hpc-work/protein-diffusion/data/cath_calpha/uncond_test_old_denoiser_no_noise_last5steps/eval_plots")
    parser.add_argument("--data_dir", type=str, required=False, default=None)
    parser.add_argument("--plot_width", type=str, required=False, default="col")
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample_df)
    ref_df = pd.read_csv(args.ref_df)
    if args.cond_df:
        cond_df = pd.read_csv(args.cond_df)
    else:
        cond_df = None
    run_plot_pipeline(sample_df, ref_df, cond_df, args.plot_dir, args.data_dir, args.plot_width)