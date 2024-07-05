import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pymongo import MongoClient
import matplotlib.pyplot as plt

from similarities.settings import *


def kde_plot(x, y, kernel, plot_3d=False, top_min=None):
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
        if top_min: 
            # Plot orange plane at top min
            yy, zz = np.meshgrid(np.linspace(*ax.get_ylim()),np.linspace(*ax.get_zlim()))
            ax.plot_surface(yy*0 + top_min, yy, zz, alpha=0.3, color='orange')

            # Plot green plane for top min integration
            xx, zz = np.meshgrid(np.linspace(top_min-0.01, x.max()), np.linspace(*ax.get_zlim()))
            zz_masked = np.minimum(zz, np.interp(xx, X[:, 0], Z[:, 0]))  # Interpolate using the center row of Z
            ax.plot_surface(xx, xx*0 + Y.min(), zz_masked, alpha=0.4, color='green')
    else: 
        plt.scatter(x, y, s=0.1, color='b')
        ax = plt.gca()
        cset = ax.contour(X, Y, Z, colors='k', levels=[0.01, 0.05, 0.1, 0.5])
        ax.clabel(cset, inline=1, fontsize=10)
        if top_min: 
            ax.vlines(top_min, *ax.get_ylim(), colors='orange')
    return ax

def kde_integrals(data_df, kde_percent=1, top_percent=0.10, x_name="mfpReg_tanimoto", y_name="diff_homo",
                  prop_abs=True, plot_kde=False, plot_3d=False, save_fig=True, top_min=None, return_kernel=False, verbose=1):
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
    print(f"KDE percent {x_name} min: {kde_data[x_name].min()}") if verbose > 2 else None
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
    print(f"{x_name} top min: {top_min}") if verbose > 2 else None
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
                    mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL):
    """
    Finds the value at a specified percentile for a given field in a MongoDB collection.

    Parameters:
    field (str): The field for which the percentile value is to be calculated.
    percentile (float): The percentile to find (between 0 and 100).
    total_docs (int, optional): Total number of documents in the collection. If None, the function will query the database to find out.
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

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
    sample_pairs_df.fillna(0, inplace=True)
    print("--> Starting KDE integral analysis.") if verbose > 0 else None
    for prop in prop_cols:
        area_df[prop] = area_df.apply(
            lambda x: kde_integrals(sample_pairs_df, kde_percent=kde_percent, top_percent=top_percent,
                                    x_name=x.sim, y_name=prop, verbose=verbose, **kwargs),
            axis=1)
        print("--> Finished KDE integral analysis for {}.".format(prop)) if verbose > 1 else None
    area_df.set_index("sim", inplace=True)
    if "diff_hl" in area_df.columns:
        area_df.sort_values("diff_hl", inplace=True)

    return area_df

def randomize_similarities(base_df, sim_metrics=None, verbose=1):
    """
    Generates a DataFrame containing the Kernel Density Estimation (KDE) integrals for specified similarity metrics and properties.

    Parameters:
    base_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.
    sim_metrics (list, optional): List of similarity metrics to consider. If None, defaults to all columns containing "Reg_" or "SCnt_" 
    verbose (int, optional): Verbosity level for logging progress and debug information. Default is 1.

    Returns:
    pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
    """
    # Identify similarity metric columns
    sim_cols = [c for c in base_df.columns if ("Reg_" in c or "SCnt_" in c)]

    num_rows = base_df.shape[0]
    for sim in sim_cols: 
        base_df[sim] = np.random.uniform(0, 1, num_rows)

    return base_df


def random_sample_nosql(x=None, y=None, size=1000, verbose=1, kde_percent=1,
                        mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL):
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
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

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


def random_kde_nosql(x="mfpReg_tanimoto", y="diff_homo", size=1000, rand_seed=1,
                     top_percent=0.10, verbose=1, return_df=False, plot_kde=False, **kwargs):
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
    sample_pairs_csv = DATA_DIR / "composite_data" / f"Combo_{size}size_{rand_seed:02d}.csv"
    if not os.path.isfile(sample_pairs_csv):
        df = random_sample_nosql(x=x, y=y, size=size, verbose=verbose, kde_percent=kde_percent, **kwargs)
    else: 
        df = pd.read_csv(sample_pairs_csv, index_col=0)
    print("Starting analysis...") if verbose else None
    perc = kde_integrals(df, x_name=x, y_name=y, kde_percent=1, top_percent=top_percent, plot_kde=plot_kde)
    return (perc, df) if return_df else perc


def rand_composite_nosql(size, kde_percent, top_percent, num_trials=30, plot=True, verbose=1, random=False, ylims=None, ax=None,
                         mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL):
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
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

    """
    avg_dfs = []
    comp_dir = DATA_DIR / "composite_data"

    # Iterate through multiple trials
    anal_name = f"{size}size_{int(kde_percent * 100):02d}kde_{int(top_percent * 100):02d}top" + ("_random" if random else "")
    for i in range(num_trials):
        trial_name = f"{anal_name}_Rand{i:02d}" + ("_random" if random else "")
        print("Creating data sample with random seed {}...".format(i)) if verbose else None

        # Check if the area_df_csv file exists
        area_df_csv = comp_dir / f"IntegralRatios_{trial_name}.csv"
        if not os.path.isfile(area_df_csv):
            # If sample_pairs_csv does not exist, generate a new random sample and save to CSV
            sample_pairs_csv = comp_dir / f"Combo_{size}size_{i:02d}.csv"
            if not os.path.isfile(sample_pairs_csv):
                # Establish one variable, _working_df, so only one DataFrame is held in memory 
                _working_df = random_sample_nosql(size=size, verbose=verbose,
                                                  mongo_uri=mongo_uri, mongo_db=mongo_db, mongo_coll=mongo_coll)
                _working_df.to_csv(sample_pairs_csv)
            else:
                _working_df = pd.read_csv(sample_pairs_csv, index_col=0)

            # Generate KDE integrals DataFrame and save to CSV
            if random: 
                random_csv = comp_dir / f"Combo_{size}size_{i:02d}_random.csv"
                if not os.path.isfile(random_csv): 
                    _working_df = randomize_similarities(_working_df)
                    _working_df.to_csv(random_csv)
                else: 
                    _working_df = pd.read_csv(random_csv, index_col=0)
            _working_df = generate_kde_df(_working_df, kde_percent=kde_percent, top_percent=top_percent, verbose=verbose)
            _working_df.to_csv(area_df_csv)
        else:
            _working_df = pd.read_csv(area_df_csv, index_col=0)

        # Append the average series of the DataFrame to avg_dfs list
        avg_dfs.append(pd.Series(_working_df.mean(axis=1)))

    # Concatenate all average series into avg_df DataFrame
    avg_df = pd.concat(avg_dfs, axis=1)

    # Sort avg_df by the maximum value in each row
    sort_value = (avg_df.mean(axis=1) + avg_df.std(axis=1)).sort_values()
    avg_df = avg_df.reindex(sort_value.index)

    # Save avg_df to CSV
    avg_df.to_csv(comp_dir / f"AvgIntegralRatios_{trial_name}.csv")

    # Plotting if plot=True
    if plot:
        mean_row, std_row = avg_df.mean(axis=1), avg_df.std(axis=1)
        ax = sns.scatterplot(avg_df, s=10, ax=ax)
        ax.plot(mean_row, label='Mean', color='blue')
        ax.fill_between(mean_row.index, mean_row - std_row, mean_row + std_row, color='blue', alpha=0.2, label='1 Std Dev')
        # ax = avg_df.plot(figsize=(10, 6))
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_xticks(range(0, len(avg_df.index)), avg_df.index, rotation="vertical", fontsize=10)
        if ylims: 
            ax.set_ylim(*ylims)
        ax.set_xlabel("Similarity Measure")
        ax.set_ylabel("Average Top Area Ratio")
        ax.set_title(anal_name.replace("_", " ").capitalize())
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"AvgIntegralRatios_{anal_name}_{num_trials:02d}trials.png", dpi=300)
        print("Done. Plots saved") if verbose else None
        return ax
    return avg_df


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


def compare_plt(x, y, df, bin_num=20, top_bin_edge=None, prop_abs=True, save=True, bin_data=True,
                x_i=None, name_tag="", **kwargs):
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
          x_i (float, optional): Lower limit for the independent variable.
          name_tag (str, optional): output file name tag

      Returns:
          None
      """
    plot_df = df[[x, y]]
    if prop_abs:
        plot_df[y] = plot_df[y].apply(abs)

    # Get x std data
    if bin_data: 
        print("Binning data. ")
        bin_df = get_bin_data(x=x, y=y, df=plot_df, bin_num=bin_num, top_bin_edge=top_bin_edge)

    # Plot
    print("Plotting data points.")
    sns.jointplot(plot_df, x=x, y=y, kind='hex', bins="log", dropna=True)
    # plt.scatter(plot_df[x], plot_df[y], alpha=0.2, marker='.')
    ax = plt.gca()
    print("Plotting std. dev. bars")
    if bin_data: 
        ax.errorbar(bin_df["intvl_mid"], bin_df["mean"], yerr=bin_df["std"], fmt=".", color='darkorange', elinewidth=2)
        ax.legend(["Compared Mols", "Fitted Boundary", "Bin Mean & Std. Dev." ])
    ax.set_xlabel("Similarity ({} finderprint / {} similarity score)".format(*x.split("_")))
    ax.set_ylabel("{}Difference in {} values".format("Absolute " if prop_abs else "", y.split("_")[1].upper()))
    if save:
        plt.savefig(PLOT_DIR / f"SinglePlt{name_tag}_{x}_{y.strip('diff_')}{'_abs' if prop_abs else ''}.png",
                    dpi=300)

    return ax


def batch_db_kde(sim="mfpReg_tanimoto", prop="diff_homo", batch_size=10000, total_docs=353314653, 
                 kde_percent=0.10, top_percent=0.10, zeros_cutoff=1e-10, 
                 verbose=2, return_all_d=False,
                 mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs"):
    """
    Perform batch Kernel Density Estimation (KDE) analysis on a dataset stored in a 
    MongoDB collection and return percent of the top integrated KDE area.

    Parameters:
    sim (str): Similarity metric to use for KDE analysis. Default is "mfpReg_tanimoto".
    prop (str): Property to analyze alongside the similarity metric. Default is "diff_homo".
    batch_size (int): Number of documents to process in each batch. Default is 10,000.
    total_docs (int): Total number of documents in the collection. Default is 353,314,653.
    kde_percent (float): Proportion of the data to use for KDE. Default is 0.10 (10%).
    top_percent (float): Top percentile of the data to focus the KDE integral analysis on. Default is 0.10 (10%).
    zeros_cutoff (float): Threshold below which the KDE percent is considered zero. Default is 1e-10.
    verbose (int): Verbosity level for logging progress and debug information. Default is 2.
    return_all_d (bool): Whether to return the detailed results DataFrame. Default is False.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is "mol_pairs".

    Returns:
    float: Percent of the KDE top area.
    (float, pd.DataFrame): If return_all_d is True, returns a tuple with the average percent and the detailed results DataFrame.
    """
    if not total_docs: 
        total_docs = collection.count_documents({})
        print("Starting total number of docs query...") if verbose > 2 else None

    # Calculate the index to skip to reach the top 10th percentile
    percentile = 100 - top_percent*kde_percent*100
    divide = find_percentile(sim, percentile=percentile, total_docs=total_docs, verbose=verbose)
    print(f"{sim} {percentile} percentile value ({top_percent*100} percentile of top {kde_percent*100}%): ", divide) if verbose > 1 else None

    # Set up batch analysis 
    anal_num = round(total_docs * kde_percent)
    total_processed = 0
    result_list = []
    perc = 1
    
    print(f"Starting analysis for {anal_num} docs...") if verbose > 1 else None
    while total_processed < anal_num:
        # Fetch the batch of documents
        current_batch_size = min(batch_size, anal_num - total_processed)
        sort_dir = -1 if (total_processed < total_docs/2) else 1
        skip_num = total_processed if (total_processed < total_docs/2) else anal_num - (total_processed - current_batch_size) 

        if perc < zeros_cutoff:  # The percents keep getting lower. So, if the previous percent is less than the divide, perc ~ 0
            perc, kernel = 0, None
        else: 
            print(f"...Starting query: sorting {sort_dir} and skipping {skip_num}...") if verbose > 2 else None
            with MongoClient(mongo_uri) as client:
                cursor = client[mongo_db][mongo_coll].find({}, {sim: 1, prop: 1}).sort({sim: sort_dir}).skip(skip_num).limit(current_batch_size)
                b_df = pd.DataFrame(list(cursor))
              
            # Apply the analysis function to the DataFrame
            d_min, d_max = b_df[sim].min(), b_df[sim].max()
            print(f"  --> DF min {d_min}, DF max {d_max}") if verbose > 2 else None
            
            if d_min == d_max:  # If the DF covers no range, the percent KDE will fail. 
                perc, kernel = 1 if d_min > divide else 0, None
            else: 
                perc, kernel = kde_integrals(b_df, x_name=sim, y_name=prop, kde_percent=1, top_min=divide, top_percent=top_percent, plot_kde=False, return_kernel=True)
        result_list.append({"percent": perc, "length": b_df.shape[0], "kernel": kernel})
        print(f"  --> Processed {total_processed} documents. Current KDE analysis ({current_batch_size} docs) yields {perc}%") if verbose > 1 else None

        # Update the counter
        processed_count = b_df.shape[0]
        total_processed += processed_count

    results = pd.DataFrame(result_list)
    results.to_csv(DATA_DIR / f"BatchKDE_{sim}_{prop}_kde{int(kde_percent*100):02d}_{int(top_percent*100):02d}top.csv")
    avg_perc = (results["percent"] * results["length"]).sum() / results["length"].sum()
    print(f"KDE Top Area for {sim} and {prop}: {avg_perc}") if verbose else None
    return (avg_perc, results) if return_all_d else avg_perc


def batch_kde_all(kde_percent=0.05, top_percent= .10, replace=False, batch_size=100000, total_docs=353314653, verbose=2):
    with MongoClient(MONGO_CONNECT) as client:
        all_cols = [k for k in client[MONGO_DB]["mol_pairs"].find_one().keys() if k not in ['_id', 'id_1', 'id_2']]
    sim_cols = [c for c in all_cols if ("Reg_" in c or "SCnt_" in c)]
    prop_cols = [c for c in all_cols if (c not in sim_cols)]
    
    save_file = DATA_DIR / f"IntegralRatios_DB_kde{int(kde_percent*100):02d}_top{int(top_percent*100):02d}.csv"
    sims = [s for s in sim_cols if "mcconnaughey" not in s]
    area_df = pd.DataFrame(index=sims, columns=prop_cols)
    
    for y in prop_cols:
        for x in sims: 
            if not replace: 
                area_df = pd.read_csv(save_file, index_col=0)
                if pd.notna(area_df.at[x,y]): 
                    print(f"--> KDE Top Area for {x} and {y} already exists: {area_df.at[x,y]}") if verbose else None
                    continue
            avg_perc = batch_db_kde(sim=x, prop=y, batch_size=batch_size, total_docs=total_docs,
                                    kde_percent=kde_percent, top_percent=top_percent, 
                                    verbose=verbose, return_all_d=False)
            print(f"--> KDE Top Area for {x} and {y}: {avg_perc}") if verbose else None
            area_df.at[x,y] = avg_perc
            area_df.to_csv(save_file)
    return area_df


def plot_area_df(area_df): 
    area_df.sort_values("diff_homo", inplace=True)
    area_df.plot(figsize=(10,6))
    plt.xticks(range(0,len(area_df.index)), area_df.index, rotation="vertical", fontsize=10)
    plt.xlabel("Similarity metric")
    plt.ylabel("Ratio of the KDE integral of the top data percentile ")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", f"IntegralRatios_kde{int(kde_percent*100):02d}_top{int(top_percent*100):02d}.png"), dpi=300)
