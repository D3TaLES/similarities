import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from generate_comparisons import create_compare_df, get_all_d

BASE_DIR = Path(__file__).resolve().parent
PERCENT = 0.05
TOP_PERCENT = 0.10
NUM_TRIALS = 30


def kde_plot(x, y, kernel):
    # Contour plot
    X, Y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    # sns.jointplot(x=x, y=y, kind='hex', bins="log", dropna=True)
    plt.scatter(x, y, s=0.1, color='b')
    ax = plt.gca()
    cset = ax.contour(X, Y, Z, colors='k', levels=[0.01, 0.05, 0.1, 0.5])
    ax.clabel(cset, inline=1, fontsize=10)
    return ax


def kde_integrals(data_df, x_name="mfpReg_tanimoto", y_name="diff_homo", data_percent=1, top_percent=0.10,
                  prop_abs=True, plot_kde=False, save_fig=True):
    """
    Compute the ratio of the kernel density estimate (KDE) integral of the top percentile data to the integral of the entire dataset.

    Args:
        data_df (DataFrame): DataFrame containing the data.
        x_name (str, optional): Name of the column representing the independent variable. Default is "mfpReg_tanimoto".
        y_name (str, optional): Name of the column representing the dependent variable. Default is "diff_homo".
        data_percent (float, optional): Percentage of the data (as decimal) to consider for KDE estimation. Default is 1.
        top_percent (float, optional): Percentage of top data (as decimal) to compare with the entire KDE. Default is 0.10.
        prop_abs (bool, optional): Whether to take the absolute value of the dependent variable. Default is True.
        plot_kde (bool, optional): Whether to plot the KDE and related information. Default is False.
        save_fig (bool, optional): Whether to save KDE plot. Default is True.

    Returns:
        float: The ratio of the KDE integral of the top percentile data to the integral of the entire dataset.

    Note:
        If `plot_kde` is True, the function will also plot the KDE contour and a vertical line representing the top percentile divide. It will save the plot as an image file.
  """
    kde_data = data_df.nlargest(round(len(data_df.index) * data_percent), x_name)
    if prop_abs:
        kde_data[y_name] = kde_data[y_name].apply(abs)
    # Perform the kernel density estimate
    x = np.array(kde_data[x_name].values.tolist())
    y = np.array(kde_data[y_name].values.tolist())
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values, bw_method=1)

    # Get top entries and compare the KDE integral of that range with whole range
    top_min = kde_data.nlargest(round(len(kde_data.index) * top_percent), x_name)[x_name].min()
    total_integ = kernel.integrate_box((0, 0), (np.inf, np.inf))  # -np.inf, -np.inf
    top_integ = kernel.integrate_box((top_min, 0), (np.inf, np.inf))
    percent_top_area = top_integ / total_integ

    # Plot data, KDE contour, and vertical line representing top percent divide
    if plot_kde:
        ax = kde_plot(x, y, kernel)
        ax.vlines(top_min, *ax.get_ylim(), colors='orange')
        ax.legend(["Comparison data", f"Top {top_percent * 100}% divide "])
        ax.set_xlabel("Similarity ({} finderprint / {} similarity score)".format(*x_name.split("_")))
        ax.set_ylabel("{}Difference in {} values".format("Absolute " if prop_abs else "", y_name.split("_")[1].upper()))
        ax.text(x.max() - 0.13, y.max() - 1, f"Area: {percent_top_area * 100:.2f}%", fontsize=14)
        if save_fig:
            plt.savefig(os.path.join(BASE_DIR, "plots",
                                     f"SingleKDEPlt{PERCENT:02d}perc_top{int(top_percent * 100):02d}_{x_name}_{y_name.strip('diff_')}{'_abs' if prop_abs else ''}.png"),
                        dpi=300)
        return ax

    return percent_top_area


def generate_kde_df(compare_df, verbose=1, top_percent=TOP_PERCENT):
    sim_cols = [c for c in compare_df.columns if ("Reg_" in c or "SCnt_" in c)]
    sims = [s for s in sim_cols if "mcconnaughey" not in s]
    prop_cols = [c for c in compare_df.columns if (c not in sim_cols and "id_" not in c)]

    area_df = pd.DataFrame(index=range(len(sims)), columns=[])
    area_df["sim"] = sims
    print("--> Starting KDE integral analysis.") if verbose > 0 else None
    for prop in prop_cols:
        area_df[prop] = area_df.apply(
            lambda x: kde_integrals(compare_df, x_name=x.sim, y_name=prop, data_percent=0.10,
                                    top_percent=top_percent, save_fig=False),
            axis=1)
        print("--> Finished KDE integral analysis for {}.".format(prop)) if verbose > 1 else None
    area_df.set_index("sim", inplace=True)
    area_df.sort_values("diff_hl", inplace=True)
    return area_df


if __name__ == "__main__":
    all_d = get_all_d()

    avg_dfs = []
    for i in range(NUM_TRIALS):
        print("Creating data sample with random seed {}...".format(i))
        compare_df_csv = os.path.join(BASE_DIR, "data_files", f"Combo_{int(PERCENT*100):02d}perc_Rand{i:02d}.csv")
        area_df_csv = os.path.join(BASE_DIR, "data_files", f"IntegralRatios_{int(PERCENT*100):02d}perc_top{int(TOP_PERCENT*100):02d}_Rand{i:02d}.csv")

        if not os.path.isfile(area_df_csv):
            if not os.path.isfile(compare_df_csv):
                working_df = create_compare_df(all_d, frac=PERCENT, random_seed=i, verbose=2)
                working_df.to_csv(compare_df_csv)
            else:
                working_df = pd.read_csv(compare_df_csv, index_col=0)
            working_df = generate_kde_df(working_df, top_percent=TOP_PERCENT, verbose=2)
            working_df.to_csv(area_df_csv)
        else:
            working_df = pd.read_csv(area_df_csv, index_col=0)
        avg_dfs.append(pd.Series(working_df.mean(axis=1)))

    avg_df = pd.concat(avg_dfs, axis=1)
    avg_df = avg_df.reindex(avg_df.max(axis=1).sort_values().index)
    avg_df.to_csv(os.path.join(BASE_DIR, "data_files", f"AvgIntegralRatios_{int(PERCENT*100):02d}perc_top{int(TOP_PERCENT*100):02d}_{NUM_TRIALS:02d}trials.csv"))

    ax = sns.scatterplot(avg_df)
    # ax = avg_df.plot(figsize=(10, 6))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(range(0, len(avg_df.index)), avg_df.index, rotation="vertical", fontsize=10)
    plt.xlabel("Similarity metric")
    plt.ylabel("Normalized average ratio of \nthe KDE integrals of the top data percentile ")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", f"AvgIntegralRatios_{int(PERCENT*100):02d}perc_top{int(TOP_PERCENT*100):02d}_{NUM_TRIALS:02d}trials.png"), dpi=300)
    print("Done. Plots saved")
