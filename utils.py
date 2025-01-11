import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

