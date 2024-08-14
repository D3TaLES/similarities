import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def diagonal_line(x, test_x, y_max=1, x2=1, y2=0):
    x1, y1 = test_x, y_max
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


def diag_test(x, y, test_x, count_backside=False, **line_kwargs):
    if (not count_backside) and (x < test_x):
        return False
    return True if diagonal_line(x, test_x, **line_kwargs) > y else False


def lrt_density(t_x, data_df, x_name="x_norm", y_name="y_norm"):
    lrt_points = data_df.apply(lambda r: diag_test(r[x_name], r[y_name], t_x), axis=1).sum()
    lrt_area = 0.5 * (1 - t_x)
    return lrt_points / lrt_area


def find_diagonal(data_df, buffer=1000, x_name="x_norm", y_name="y_norm", return_dict=False):
    dens_dict = {x: 0 for x in np.arange(0, 1, 0.001)}
    for x in tqdm.tqdm(dens_dict.keys()):
        dens = lrt_density(x, data_df, x_name=x_name, y_name=y_name)
        dens_dict[x] = dens
        max_dens_x = max(dens_dict, key=dens_dict.get)
        if dens < (dens_dict[max_dens_x] - buffer):
            return dens_dict if return_dict else max_dens_x


def neighborhood_ratio(data_df, x_name="mfpReg_tanimoto", y_name="diff_homo", plot=False):
    # Normalize x and y data
    df_i = data_df[[x_name, y_name]].copy()
    df_i["x_norm"] = MinMaxScaler().fit_transform(np.array(data_df[x_name]).reshape(-1, 1))
    df_i["y_norm"] = MinMaxScaler().fit_transform(np.array(data_df[y_name].abs()).reshape(-1, 1))

    # Find diagonal line and compute point densities.
    t_x = find_diagonal(df_i, buffer=100, x_name="x_norm", y_name="y_norm", return_dict=False)
    neighborhood_points = df_i.apply(lambda r: diag_test(r.x_norm, r.y_norm, t_x, count_backside=True), axis=1).sum()
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
