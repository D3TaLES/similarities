import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from similarities.generation import create_compare_df, get_all_d

from similarities.settings import *


# Plot Function
def quad_f(x, a, h, k):
    """
    Return a quadratic function.
    """
    return -a * (x - h) ** 2 + k


# Define the residual function
def residual(params, data, penalty=15, func=quad_f):
    """
    Calculate the residual error for fitting.

    Args:
        params (array_like): Parameters for the function.
        data (array_like): Data points to fit.
        penalty (float, optional): Penalty coefficient for error calculation.
        func (function, optional): Function to fit the data.

    Returns:
        float: Total residual error.
    """

    total_error = 0
    for [x, y] in data:
        fy = func(x, *params)
        if y <= fy:
            total_error += (fy - y) ** 2
        else:
            total_error += np.exp(penalty * (abs(fy - y)))
    return total_error


def get_boundary(data_df, x, y, x_i=0.4, y_i=None, penalty=15, func=quad_f, a_bounds=(None, None),
                 h_bounds=(None, None), k_bounds=(None, None)):
    """
    Fit boundary for data points.

    Args:
        data_df (DataFrame): Dataframe containing the data.
        x (str): Name of the column representing the independent variable.
        y (str): Name of the column representing the dependent variable.
        x_i (float, optional): Lower limit for the independent variable.
        y_i (float, optional): Upper limit for the dependent variable.
        penalty (float, optional): Penalty coefficient for error calculation.
        func (function, optional): Function to fit the data.

    Returns:
        OptimizeResult: Optimization result.
    """
    # Define the data
    if y_i:
        cut_data = data_df[data_df.apply(lambda r: r[y] >= -(y_i / x_i) * r[x] + y_i, axis=1)]
    else:
        cut_data = data_df[data_df[x] > x_i]
    datatofit = np.array(cut_data.values.tolist())

    # Perform nonlinear fitting
    result = minimize(residual, x0=[0, 0, 0], args=(datatofit, penalty, func), bounds=(a_bounds, h_bounds, k_bounds))
    return result


def get_bin_data(x, y, df, bin_num=20, top_bin_edge=None):
    """
      Get binned data for plotting.

      Args:
          x (str): Name of the column representing the independent variable.
          y (str): Name of the column representing the dependent variable.
          df (DataFrame): Dataframe containing the data.
          bin_num (int, optional): Number of bins.
          top_bin_edge (float, optional): Upper limit for the bin edges.

      Returns:
          DataFrame: Dataframe containing binned data.
      """
    bin_intvs = pd.cut(df[x], bins=bin_num)
    if top_bin_edge:
        max_x = float(df[x].max())
        bin_intvs = bin_intvs.apply(lambda x: x if x.left <= top_bin_edge else pd.Interval(top_bin_edge, max_x))
    sim_groups = df.groupby(bin_intvs)
    bin_df = sim_groups[y].agg(["count", "mean", "std"])
    bin_df['intvl_mid'] = bin_df.apply(lambda x: x.name.mid, axis=1)
    return bin_df


def compare_plt(x, y, df, bin_num=20, top_bin_edge=None, prop_abs=True, save=True, boundary_func=None,
                x_i=None, name_tag="",**kwargs):
    """
      Compare and plot data.

      Args:
          x (str): Name of the column representing the independent variable.
          y (str): Name of the column representing the dependent variable.
          df (DataFrame, optional): Dataframe containing the data.
          bin_num (int, optional): Number of bins.
          top_bin_edge (float, optional): Upper limit for the bin edges.
          prop_abs (bool, optional): Whether to use the absolute value of the dependent variable.
          save (bool, optional): Whether to save the plot.
          boundary_func (function, optional): Function to fit the boundary.
          x_i (float, optional): Lower limit for the independent variable.
          name_tag (str, optional): output file name tag

      Returns:
          None
      """
    plot_df = df[[x, y]]
    if prop_abs:
        plot_df[y] = plot_df[y].apply(abs)

    # Get x std data
    print("Binning data. ")
    bin_df = get_bin_data(x=x, y=y, df=plot_df, bin_num=bin_num, top_bin_edge=top_bin_edge)

    # Plot
    print("Plotting data points.")
    sns.jointplot(plot_df, x=x, y=y, kind='hex', bins="log", dropna=True)
    # plt.scatter(plot_df[x], plot_df[y], alpha=0.2, marker='.')
    ax = plt.gca()
    print("Plotting std. dev. bars")
    ax.errorbar(bin_df["intvl_mid"], bin_df["mean"], yerr=bin_df["std"], fmt=".", color='darkorange', elinewidth=2)
    if boundary_func:
        print("Calculating boundary function. ")
        # Plot the data and fitted model
        x_i = x_i or round(plot_df[x].mean() + plot_df[x].std(), 2)
        k = plot_df[y].max() * 1.01
        print("Xi: ", x_i)
        result = get_boundary(plot_df, x=x, y=y, func=boundary_func, x_i=x_i, a_bounds=(0, None), h_bounds=(None, 0),
                              k_bounds=(None, None), **kwargs)
        x_values = np.linspace(x_i, plot_df[x].max(), 100)
        ax.plot(x_values, quad_f(x_values, *result.x), 'k-', label='Fitted Model')

    ax.legend(["Compared Mols", "Fitted Boundary", "Bin Mean & Std. Dev."])
    ax.set_xlabel("Similarity ({} finderprint / {} similarity score)".format(*x.split("_")))
    ax.set_xlabel("{}Difference in {} values".format("Absolute " if prop_abs else "", y.split("_")[1].upper()))
    if save:
        plt.savefig(os.path.join(BASE_DIR, "plots",
                                 f"SinglePlt{name_tag}_{x}_{y.strip('diff_')}{'_abs' if prop_abs else ''}.png"),
                    dpi=300)

    return ax


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
                                     f"SingleKDEPlt{int(data_percent*100):02d}perc_top{int(top_percent * 100):02d}_{x_name}_{y_name.strip('diff_')}{'_abs' if prop_abs else ''}.png"),
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


def rand_composite_analysis(all_df, num_trials=NUM_TRIALS, data_percent=DATA_PERCENT, top_percent=TOP_PERCENT, plot=True):
    avg_dfs = []
    for i in range(num_trials):
        print("Creating data sample with random seed {}...".format(i))
        compare_df_csv = os.path.join(BASE_DIR, "composite_data",
                                      f"Combo_{int(data_percent * 100):02d}perc_Rand{i:02d}.csv")
        area_df_csv = os.path.join(BASE_DIR, "composite_data",
                                   f"IntegralRatios_{int(data_percent * 100):02d}perc_top{int(top_percent * 100):02d}_Rand{i:02d}.csv")

        if not os.path.isfile(area_df_csv):
            if not os.path.isfile(compare_df_csv):
                working_df = create_compare_df(all_df, frac=data_percent, random_seed=i, verbose=2)
                working_df.to_csv(compare_df_csv)
            else:
                working_df = pd.read_csv(compare_df_csv, index_col=0)
            working_df = generate_kde_df(working_df, top_percent=top_percent, verbose=2)
            working_df.to_csv(area_df_csv)
        else:
            working_df = pd.read_csv(area_df_csv, index_col=0)
        avg_dfs.append(pd.Series(working_df.mean(axis=1)))

    avg_df = pd.concat(avg_dfs, axis=1)
    avg_df = avg_df.reindex(avg_df.max(axis=1).sort_values().index)
    avg_df.to_csv(os.path.join(BASE_DIR, "composite_data",
                               f"AvgIntegralRatios_{int(data_percent * 100):02d}perc_top{int(top_percent * 100):02d}_{num_trials:02d}trials.csv"))

    if plot:
        ax = sns.scatterplot(avg_df)
        # ax = avg_df.plot(figsize=(10, 6))
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.xticks(range(0, len(avg_df.index)), avg_df.index, rotation="vertical", fontsize=10)
        plt.xlabel("Similarity metric")
        plt.ylabel("Normalized average ratio of \nthe KDE integrals of the top data percentile ")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "plots",
                                 f"AvgIntegralRatios_{int(data_percent * 100):02d}perc_top{int(top_percent * 100):02d}_{num_trials:02d}trials.png"),
                    dpi=300)
        print("Done. Plots saved")


if __name__ == "__main__":
    all_d = get_all_d()

    # # Composite analysis
    # rand_composite_analysis(all_d, num_trials=NUM_TRIALS, data_percent=DATA_PERCENT, top_percent=TOP_PERCENT, plot=True)

    # Comparison DF
    compare_df = pd.read_csv(os.path.join(BASE_DIR, "composite_data", f"combo_sims_{percent:02d}perc.csv"), index_col=0)
    sim_reg_cols = [c for c in compare_df.columns if "Reg_" in c]
    sim_scnt_cols = [c for c in compare_df.columns if "SCnt_" in c]
    sim_cols = sim_reg_cols + sim_scnt_cols
    prop_cols = [c for c in compare_df.columns if (c not in sim_cols and "id_" not in c)]
    print("Num Instances: ", compare_df.shape[0])

    # print("Generating master regular fingerprint plots...")
    # reg_plts = sns.pairplot(compare_df, kind="hist", x_vars=sim_reg_cols, y_vars=prop_cols, dropna=True)
    # reg_plts.savefig(os.path.join(BASE_DIR, "plots", f"Sims-ElecProps_Reg_{percent:02d}perc.png"), dpi=300)
    #
    # print("Generating master simulated count fingerprint plots...")
    # scnt_plts = sns.pairplot(compare_df, kind="hist", x_vars=sim_scnt_cols, y_vars=prop_cols, dropna=True)
    # scnt_plts.savefig(os.path.join(BASE_DIR, "plots", f"Sims-ElecProps_SCnt_{percent:02d}perc.png"), dpi=300)

    compare_plt("ttrReg_cosine", "diff_homo", compare_df, top_bin_edge=0.8, boundary_func=quad_f, penalty=15,
                name_tag=f"{DATA_PERCENT:02d}perc")