import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pymongo import MongoClient
import matplotlib.pyplot as plt

from similarities.settings import *


def kde_plot(x, y, kernel, plot_3d=False):
    """
    Creates a plot of Kernel Density Estimation (KDE) for given data.

    Parameters:
    x (array-like): 1D array of x-values of the data points.
    y (array-like): 1D array of y-values of the data points.
    kernel (callable): KDE function that computes the density estimate.
                       This is typically generated using a library like scipy.stats.gaussian_kde.
    plot_3d (bool, optional): If True, creates a 3D surface plot of the KDE.
                              If False, creates a 2D contour plot. Default is False.

    Returns:
    matplotlib.axes._axes.Axes: The matplotlib axes object with the plot.
    """
    # Contour plot
    X, Y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    # sns.jointplot(x=x, y=y, kind='hex', bins="log", dropna=True)
    if plot_3d: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="Blues", lw=0.1, rstride=1, cstride=1, ec='k')
        ax.set_zlabel("KDE", fontsize=10, rotation=90)
    else: 
        plt.scatter(x, y, s=0.1, color='b')
        ax = plt.gca()
        cset = ax.contour(X, Y, Z, colors='k', levels=[0.01, 0.05, 0.1, 0.5])
        ax.clabel(cset, inline=1, fontsize=10)
    return ax


def kde_integrals(data_df, kde_percent=1, top_percent=0.10, x_name="mfpReg_tanimoto", y_name="diff_homo",
                  prop_abs=True, plot_kde=False, plot_3d=False, save_fig=True, top_min=None, return_kernel=False):
    """
    Compute the ratio of the kernel density estimate (KDE) integral of the top percentile data to the integral of the entire dataset.

    Args:
        data_df (DataFrame): DataFrame containing the data.
        kde_percent (float, optional): Percentage of the data (as decimal) to consider for KDE estimation. Default is 1.
        top_percent (float, optional): Percentage of top data (as decimal) to compare with the entire KDE. Default is 0.10.
        x_name (str, optional): Name of the column representing the independent variable. Default is "mfpReg_tanimoto".
        y_name (str, optional): Name of the column representing the dependent variable. Default is "diff_homo".
        prop_abs (bool, optional): Whether to take the absolute value of the dependent variable. Default is True.
        plot_kde (bool, optional): Whether to plot the KDE and related information. Default is False.
        save_fig (bool, optional): Whether to save KDE plot. Default is True.
        top_min (float, optional): Manual value for minimum value to compare with entire KDE. Default is None. 

    Returns:
        float: The ratio of the KDE integral of the top percentile data to the integral of the entire dataset.

    Note:
        If `plot_kde` is True, the function will also plot the KDE contour and a vertical line representing the top percentile divide. It will save the plot as an image file.
  """
    kde_data = data_df.nlargest(round(len(data_df.index) * kde_percent), x_name)
    if prop_abs:
        kde_data[y_name] = kde_data[y_name].apply(abs)
    # Perform the kernel density estimate
    x = np.array(kde_data[x_name].values.tolist())
    y = np.array(kde_data[y_name].values.tolist())
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values, bw_method=1)

    # Get top entries and compare the KDE integral of that range with whole range
    if not top_min: 
        top_min = kde_data.nlargest(round(len(kde_data.index) * top_percent), x_name)[x_name].min()
    total_integ = kernel.integrate_box((0, 0), (np.inf, np.inf))  # -np.inf, -np.inf
    top_integ = kernel.integrate_box((top_min, 0), (np.inf, np.inf))
    percent_top_area = top_integ / total_integ

    # Plot data, KDE contour, and vertical line representing top percent divide
    if plot_kde or plot_3d:
        ax = kde_plot(x, y, kernel, plot_3d=plot_3d)
        if plot_3d: 
            yy, zz = np.meshgrid(np.linspace(*ax.get_ylim()),np.linspace(*ax.get_zlim()))
            ax.plot_surface(yy*0 + top_min, yy, zz, alpha=0.3, color='orange')
        else:
            ax.vlines(top_min, *ax.get_ylim(), colors='orange')
        ax.legend(["Comparison data", f"Top {top_percent * 100}% divide "])
        ax.set_xlabel("Similarity ({} fingerprint / \n{} similarity score)".format(*x_name.split("_")))
        ax.set_ylabel("{}Difference in \n{} values".format("Absolute " if prop_abs else "", y_name.split("_")[1].upper()))
        # ax.text(x.max() - 0.13, y.max() - 1, f"Area: {percent_top_area * 100:.2f}%", fontsize=14)
        if save_fig:
            plt.savefig(PLOT_DIR /f"{'3D' if plot_3d else ''}SingleKDEPlt{int(kde_percent*100):02d}perc_top{int(top_percent * 100):02d}"
                                  f"_{x_name}_{y_name.strip('diff_')}{'_abs' if prop_abs else ''}.png",
                        dpi=300)
    return (percent_top_area, kernel) if return_kernel else percent_top_area


def find_percentile(field, percentile, total_docs=None, verbose=1,
                    mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs"):
    """
    Finds the value at a specified percentile for a given field in a MongoDB collection.

    Parameters:
    field (str): The field for which the percentile value is to be calculated.
    percentile (float): The percentile to find (between 0 and 100).
    total_docs (int, optional): Total number of documents in the collection. If None, the function will query the database to find out.
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is "mol_pairs".

    Returns:
    float: The value at the specified percentile for the given field.
    """
    if not total_docs: 
        print("Starting total number of docs query...") if verbose else None
        with MongoClient(mongo_uri) as client:
            total_docs = client[mongo_db][mongo_coll].count_documents({})
    if percentile < 50: 
        sort_dir = 1 
        percentile_idx = int(total_docs * (percentile/100))
    else: 
        sort_dir = -1 
        percentile_idx = int(total_docs * ((100-percentile)/100))
    print(f"Percentile index {percentile_idx} out of total {total_docs} documents.") if verbose else None

    print("Starting sort and skip query..") if verbose else None
    with MongoClient(mongo_uri) as client:
        cursor = client[mongo_db][mongo_coll].find({}, {field: 1, '_id': 0}).sort(field, sort_dir).skip(percentile_idx).limit(1)
        return list(cursor)[0][field]


def generate_kde_df(sample_pairs_df, kde_percent, top_percent, sim_metrics=None, props=None, verbose=1, **kwargs):
    """
    Generates a DataFrame containing the Kernel Density Estimation (KDE) integrals for specified similarity metrics and properties.

    Parameters:
    sample_pairs_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.
    kde_percent (float): Proportion of the data to use for KDE analysis.
    top_percent (float): Top percentile of the data to focus the KDE analysis on.
    sim_metrics (list, optional): List of similarity metrics to consider. If None, defaults to all columns containing "Reg_" or "SCnt_" except those containing "mcconnaughey".
    props (list, optional): List of properties to consider. If None, defaults to all columns not in sim_metrics and not containing "id_".
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.
    **kwargs: Additional keyword arguments to pass to the `kde_integrals` function.

    Returns:
    pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
    """
    # Identify similarity metric columns
    sim_cols = [c for c in sample_pairs_df.columns if ("Reg_" in c or "SCnt_" in c)]
    sims = sim_metrics or [s for s in sim_cols if "mcconnaughey" not in s]

    # Identify property columns
    prop_cols = props or [c for c in sample_pairs_df.columns if (c not in sim_cols and "id_" not in c)]

    # Initialize DataFrame to store KDE integral results
    area_df = pd.DataFrame(index=range(len(sims)), columns=[])
    area_df["sim"] = sims
    print("--> Starting KDE integral analysis.") if verbose > 0 else None
    for prop in prop_cols:
        area_df[prop] = area_df.apply(
            lambda x: kde_integrals(sample_pairs_df, kde_percent=kde_percent, top_percent=top_percent,
                                    x_name=x.sim, y_name=prop, **kwargs),
            axis=1)
        print("--> Finished KDE integral analysis for {}.".format(prop)) if verbose > 1 else None
    area_df.set_index("sim", inplace=True)
    if "diff_hl" in area_df.columns:
        area_df.sort_values("diff_hl", inplace=True)

    return area_df


def random_sample_nosql(x=None, y=None, size=1000, verbose=1, kde_percent=1,
                        mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs"):
    """
    Retrieves a random sample of documents from a MongoDB collection, with options to sort, limit, and project specific fields.

    Parameters:
    x (str, optional): Field to sort by in descending order. Default is None.
    y (str, optional): Additional field to include in the projection. Default is None.
    size (int, optional): Number of documents to sample. Default is 1000.
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.
    kde_percent (float, optional): Proportion of sampled documents to include in the final output. Default is 1.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is "mol_pairs".

    Returns:
    pd.DataFrame: DataFrame containing the sampled documents, indexed by MongoDB's `_id` field.
    """
    # Create the aggregation pipeline for MongoDB
    pipeline = [{"$sample": {"size": size}}]  # Randomly select 'size' number of documents
    pipeline.append({"$sort": {x: -1}}) if x else None  # Sort by the specified field in ascending order
    pipeline.append(
        {"$limit": size * kde_percent}) if kde_percent else None  # Limit to the top 'num_top_docs' documents
    pipeline.append({'$project': {v: 1 for v in [x, y] if v}}) if (
                x or y) else None  # Include only the fields 'x' and 'y'

    with MongoClient(mongo_uri) as client:
        print(f"Starting query to select top {size * kde_percent} of {size} random docs...") if verbose else None
        results = list(client[mongo_db][mongo_coll].aggregate(pipeline))
        df = pd.DataFrame(results).set_index("_id")

    return df


def random_kde_nosql(x="mfpReg_tanimoto", y="diff_homo", size=1000,
                     top_percent=0.10, verbose=1, return_df=False, **kwargs):
    """
    Samples a random subset of documents from a MongoDB collection, performs KDE analysis, and returns the top percentile area.

    Parameters:
    x (str, optional): The field to be used as the x-axis for KDE analysis. Default is "mfpReg_tanimoto".
    y (str, optional): The field to be used as the y-axis for KDE analysis. Default is "diff_homo".
    size (int, optional): Number of documents to sample. Default is 1000.
    top_percent (float, optional): The top percentile of the data to focus the KDE analysis on. Default is 0.10 (top 10%).
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.
    return_df (bool, optional): If True, returns both the KDE top percentile area and the sampled DataFrame. Default is False.
    **kwargs: Additional keyword arguments to pass to the `random_sample_nosql` function.

    Returns:
    float or tuple: The KDE top percentile area. If `return_df` is True, returns a tuple (KDE top percentile area, DataFrame).
    """
    df = random_sample_nosql(x=x, y=y, size=size, verbose=verbose, **kwargs)
    print("Starting analysis...") if verbose else None
    perc = kde_integrals(df, x_name=x, y_name=y, kde_percent=1, top_percent=top_percent, plot_kde=False)
    return (perc, df) if return_df else perc


def rand_composite_nosql(size, kde_percent, top_percent, num_trials=30, plot=True, verbose=1,
                         mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs"):
    """
    Performs a composite analysis by sampling multiple datasets, applying KDE analysis, and aggregating results.

    Parameters:
    size (int): Number of documents to sample for each trial.
    kde_percent (float): Proportion of the data to use for KDE analysis.
    top_percent (float): Top percentile of the data to focus the KDE analysis on.
    num_trials (int, optional): Number of trials to run. Default is 30.
    plot (bool, optional): If True, plots the results. Default is True.
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is "mol_pairs".

    """
    avg_dfs = []
    comp_dir = DATA_DIR / "composite_data"

    # Iterate through multiple trials
    for i in range(num_trials):
        print("Creating data sample with random seed {}...".format(i)) if verbose else None
        sample_pairs_csv = comp_dir / f"Combo_{size}size_{i:02d}.csv"
        area_df_csv = comp_dir / f"IntegralRatios_{size}size_{int(top_percent * 100):02d}_Rand{i:02d}.csv"

        # Check if the area_df_csv file exists
        if not os.path.isfile(area_df_csv):
            # If sample_pairs_csv does not exist, generate a new random sample and save to CSV
            if not os.path.isfile(sample_pairs_csv):
                _working_df = random_sample_nosql(size=size, verbose=verbose,
                                                  mongo_uri=mongo_uri, mongo_db=mongo_db, mongo_coll=mongo_coll)
                _working_df.to_csv(sample_pairs_csv)
            else:
                _working_df = pd.read_csv(sample_pairs_csv, index_col=0)

            # Generate KDE integrals DataFrame and save to CSV
            _working_df = generate_kde_df(_working_df, kde_percent=kde_percent, top_percent=top_percent, verbose=2)
            _working_df.to_csv(area_df_csv)
        else:
            _working_df = pd.read_csv(area_df_csv, index_col=0)

        # Append the average series of the DataFrame to avg_dfs list
        avg_dfs.append(pd.Series(_working_df.mean(axis=1)))

    # Concatenate all average series into avg_df DataFrame
    avg_df = pd.concat(avg_dfs, axis=1)

    # Sort avg_df by the maximum value in each row
    avg_df = avg_df.reindex(avg_df.max(axis=1).sort_values().index)

    # Save avg_df to CSV
    avg_df.to_csv(comp_dir / f"AvgIntegralRatios_{size}size_{int(top_percent * 100):02d}_{num_trials:02d}trials.csv")

    # Plotting if plot=True
    if plot:
        ax = sns.scatterplot(avg_df)
        # ax = avg_df.plot(figsize=(10, 6))
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.xticks(range(0, len(avg_df.index)), avg_df.index, rotation="vertical", fontsize=10)
        plt.xlabel("Similarity metric")
        plt.ylabel("Normalized average ratio of \nthe KDE integrals of the top data percentile ")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"AvgIntegralRatios_{size}size_{int(top_percent * 100):02d}"
                               f"_{num_trials:02d}trials.png", dpi=300)
        print("Done. Plots saved") if verbose else None
