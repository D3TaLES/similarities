import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
        plt.savefig(PLOT_DIR / f"SinglePlt{name_tag}_{x}_{y.strip('diff_')}{'_abs' if prop_abs else ''}.png",
                    dpi=300)

    return ax



if __name__ == "__main__":
    d_percent, a_percent, t_percent = 0.10, 0.10, 0.10

    # all_d = get_all_d()

    # Comparison DF
    compare_df = pd.read_csv(DATA_DIR / f"combo_sims_{int(d_percent*100):02d}perc.csv", index_col=0)
    sim_reg_cols = [c for c in compare_df.columns if "Reg_" in c]
    sim_scnt_cols = [c for c in compare_df.columns if "SCnt_" in c]
    sim_cols = sim_reg_cols + sim_scnt_cols
    prop_cols = [c for c in compare_df.columns if (c not in sim_cols and "id_" not in c)]
    print("Num Instances: ", compare_df.shape[0])

    compare_plt("ttrReg_cosine", "diff_homo", compare_df, top_bin_edge=0.8, boundary_func=quad_f, penalty=15,
                name_tag=f"{d_percent:02d}perc")