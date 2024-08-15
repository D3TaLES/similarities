import tqdm
import swifter
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def diagonal_line(x, test_x, y_max=1, x2=1, y2=0):
    x1, y1 = test_x, y_max
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


def diag_test(x, y, test_x, count_backside=False, **line_kwargs):
    """
    Determines if a point (x, y) is below a diagonal line in a 2D plot.

    Parameters:
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - test_x (float): The x-coordinate of the reference point on the diagonal line.
    - count_backside (bool, optional): If False, points with x < test_x are not considered. Default is False.
    - **line_kwargs: Additional keyword arguments to pass to the diagonal_line function.

    Returns:
    - bool: True if the point (x, y) is below the diagonal line, otherwise False.
    """
    if (not count_backside) and (x < test_x):
        return False
    return True if y < diagonal_line(x, test_x, **line_kwargs) else False


def lrt_density(t_x, data_df, x_name="x_norm", y_name="y_norm"):
    """
    Calculates the density of points below the diagonal line in a specified region of a 2D plot.

    Parameters:
    - t_x (float): The x-coordinate of the reference point on the diagonal line.
    - data_df (DataFrame): The DataFrame containing the data points.
    - x_name (str, optional): The name of the column representing the x-coordinates. Default is "x_norm".
    - y_name (str, optional): The name of the column representing the y-coordinates. Default is "y_norm".

    Returns:
    - float: The density of points below the diagonal line.
    """
    lrt_points = data_df.apply(lambda r: diag_test(r[x_name], r[y_name], t_x), axis=1).sum()
    lrt_area = 0.5 * (1 - t_x) + (
                t_x * 1)  # area of triangle plus area of rectangle to left of triangle (area of trapezoid)
    return lrt_points / lrt_area


def find_diagonal(data_df, x_name="x_norm", y_name="y_norm"):
    """
    Finds the x-coordinate of the diagonal line that maximizes the density of points below it
    using Brent's method for optimization.

    Parameters:
    - data_df (DataFrame): The dataset containing the points to be tested.
    - x_name (str, optional): The column name for x-coordinates in `data_df`. Default is "x_norm".
    - y_name (str, optional): The column name for y-coordinates in `data_df`. Default is "y_norm".

    Returns:
    - float: The x-coordinate of the diagonal line that maximizes the point density below it.
    """

    # Define the objective function to minimize (negative density)
    def objective(t_x):
        return -lrt_density(t_x, data_df, x_name=x_name, y_name=y_name)

    # Use Brent's method to find the optimal t_x that maximizes the density
    result = opt.minimize_scalar(objective, bounds=(0, 1), method='bounded')

    return result.x


def neighborhood_ratio(data_df, x_name="mfpReg_tanimoto", y_name="diff_homo", plot=False):
    """
    Calculates the neighborhood ratio, a metric that compares the density of points in a specified region of a 2D plot
    to the overall density of points. From Journal of Medicinal Chemistry, 1996, Vol. 39, No. 16

    Parameters:
    - data_df (DataFrame): The DataFrame containing the data points.
    - x_name (str, optional): The name of the column representing the x-coordinates. Default is "mfpReg_tanimoto".
    - y_name (str, optional): The name of the column representing the y-coordinates. Default is "diff_homo".
    - plot (bool, optional): If True, plots the data points and the diagonal line. Default is False.

    Returns:
    - float: The neighborhood ratio.
    """
    # Normalize x and y data
    df_i = data_df[[x_name, y_name]].copy()
    df_i["x_norm"] = MinMaxScaler().fit_transform(np.array(data_df[x_name]).reshape(-1, 1))
    df_i["y_norm"] = MinMaxScaler().fit_transform(np.array(data_df[y_name].abs()).reshape(-1, 1))

    # Find diagonal line and compute point densities.
    t_x = find_diagonal(df_i, x_name="x_norm", y_name="y_norm")
    neighborhood_points = df_i.swifter.apply(lambda r: diag_test(r.x_norm, r.y_norm, t_x, count_backside=True), axis=1).sum()
    lrt_area = 0.5 * (1 - t_x)
    nbh_ratio = (neighborhood_points / lrt_area) / (df_i.shape[0] / 1)

    # Plot data and line
    if plot:
        df_i.plot.scatter(x="x_norm", y="y_norm")
        plt.plot([t_x, 1], [1, 0], color="r")
        plt.xlabel("Normalized {}".format(x_name))
        plt.ylabel("Normalized {}".format(y_name))
        plt.text(x=1, y=1, s="Neighborhood Ratio: {:.2f}".format(nbh_ratio), color="k", fontsize=12,
                 verticalalignment='top', horizontalalignment='right')
    return nbh_ratio
