import os
import random
import pymongo
import numpy as np
import pandas as pd
from math import comb
import seaborn as sns
from rdkit import Chem
from scipy import stats
from pymongo import MongoClient
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.preprocessing import MinMaxScaler
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from similarities.settings import *


class SimilarityAnalysisBase:
    def __init__(self, anal_percent: float = 1, top_percent: float = 1,
                 verbose: int = 3, anal_name: str = "SimAnalysis", replace_files=False,
                 elec_props: list = ELEC_PROPS, sim_metrics: dict = SIM_METRICS, fp_gens: dict = FP_GENS,
                 default_sim="mfpReg_tanimoto", default_prop="diff_homo"):
        """

        Parameters:
        anal_percent (float): Top percent of data to use in the analysis
        top_percent (float): Top percent of the data to use to calculate the top area ratio
        total_docs (int): Total number of molecule pair data entries
        verbose (int): Verbosity level (0 = silent, 1 = minimal output, 2 = detailed output).
        elec_props (list): List of electronic properties to calculate differences.
        sim_metrics (dict): Dictionary of similarity metrics to calculate.
        fp_gens (dict): Dictionary of fingerprint generation methods.
        mongo_uri (str): MongoDB connection URI.
        mongo_db (str): MongoDB database name.
        mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

        """
        self.total_docs = self._get_total_docs()
        self.verbose = verbose
        self.replace_files = replace_files

        self.anal_percent = anal_percent / 100 if anal_percent > 1 else anal_percent
        self.top_percent = top_percent / 100 if top_percent > 1 else top_percent
        self.percentile = 100 - (self.top_percent * self.anal_percent * 100)
        self.perc_name = f"kde{int(self.anal_percent * 100):02d}_{int(self.top_percent * 100):02d}top"
        self.prop_cols = [f"diff_{ep}" for ep in elec_props]
        self.sim_cols = [f"{fp}_{s.lower()}" for fp in fp_gens.keys() for s in sim_metrics.keys()]
        self.sim_metrics = sim_metrics
        self.elec_props = elec_props
        self.fp_dict = fp_gens

        self.data_dir = DATA_DIR / anal_name
        os.makedirs(self.data_dir, exist_ok=True) 
        os.makedirs(self.data_dir / "random_sampling" , exist_ok=True)
        self.plot_dir = PLOT_DIR / anal_name
        os.makedirs(self.plot_dir, exist_ok=True)
        self.batch_kde_file = self.data_dir / f"IntegralRatios_allDB_{self.perc_name}.csv"
        self.divides_file = self.data_dir / f"TopDivides_DB_percentile{int(self.percentile):02d}.csv"

        self.default_sim = default_sim
        self.default_prop = default_prop

    @staticmethod
    def kde_integrals(data_df, anal_percent=1, top_percent=0.10, x_name="mfpReg_tanimoto", y_name="diff_homo",
                      bw_method=1,
                      top_min=None, prop_abs=True, plot_kde=False, plot_3d=False, save_fig=True, plot_dir=None,
                      verbose=2, return_kernel=False):
        """
        Compute the ratio of the kernel density estimate (KDE) integral of the top percentile data to the integral of the entire dataset.

        Args:
            data_df (DataFrame): DataFrame containing the data.
            anal_percent (float, optional): Percentage of the data (as decimal) to consider for KDE estimation. Default is 1.
            top_percent (float, optional): Percentage of top data (as decimal) to compare with the entire KDE. Default is 0.10.
            x_name (str, optional): Name of the column representing the independent variable. Default is "mfpReg_tanimoto".
            y_name (str, optional): Name of the column representing the dependent variable. Default is "diff_homo".
            bw_method (float, optional): bw_method for the SciPy gaussian_kde function
            top_min (float, optional): Manual value for minimum value to compare with entire KDE. Default is None.
            prop_abs (bool, optional): Whether to take the absolute value of the dependent variable. Default is True.
            plot_kde (bool, optional): Whether to plot the KDE and related information. Default is False.
            plot_3d (bool, optional): Whether to plot the KDE in 3D. Default is False.
            save_fig (bool, optional): Whether to save KDE plot. Default is True.
            plot_dir (str, optional): Directory in which to save the plot. Default is None.
            verbose (int, optional): Verbosity level (0 = silent, 1 = minimal output, 2 = detailed output).
            return_kernel (bool, optional): Whether to return the kernel. Default is False.

        Returns:
            float: The ratio of the KDE integral of the top percentile data to the integral of the entire dataset.

        Note:
            If `plot_kde` is True, the function will also plot the KDE contour and a vertical line representing the top percentile divide. It will save the plot as an image file.
      """
        data_df.sort_values(by=[x_name, y_name], ascending=[False, True], inplace=True)
        kde_data = data_df[:int(len(data_df.index) * anal_percent)]
        print(f"KDE percent {x_name} min: {kde_data[x_name].min()}") if verbose > 2 else None
        if prop_abs:
            kde_data.loc[:, y_name] = kde_data[y_name].apply(abs)
        # Perform the kernel density estimate
        x = np.array(kde_data[x_name].values.tolist())
        y = np.array(kde_data[y_name].values.tolist())
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values, bw_method=bw_method)

        # Get top entries and compare the KDE integral of that range with whole range
        if not top_min:
            top_min = kde_data[:int(len(kde_data.index) * top_percent)][x_name].min()
        print(f"{x_name} top min: {top_min}") if verbose > 2 else None
        total_integ = kernel.integrate_box((0, 0), (np.inf, np.inf))  # -np.inf, -np.inf
        top_integ = kernel.integrate_box((top_min, 0), (np.inf, np.inf))
        percent_top_area = top_integ / total_integ

        # Plot data, KDE contour, and vertical line representing top percent divide
        if plot_kde or plot_3d:
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
                    yy, zz = np.meshgrid(np.linspace(*ax.get_ylim()), np.linspace(*ax.get_zlim()))
                    ax.plot_surface(yy * 0 + top_min, yy, zz, alpha=0.3, color='orange')

                    # Plot green plane for top min integration
                    xx, zz = np.meshgrid(np.linspace(top_min - 0.01, x.max()), np.linspace(*ax.get_zlim()))
                    zz_masked = np.minimum(zz, np.interp(xx, X[:, 0], Z[:, 0]))  # Interpolate using the center row of Z
                    ax.plot_surface(xx, xx * 0 + Y.min(), zz_masked, alpha=0.4, color='green')

                yy, zz = np.meshgrid(np.linspace(*ax.get_ylim()), np.linspace(*ax.get_zlim()))
                ax.plot_surface(yy * 0 + top_min, yy, zz, alpha=0.3, color='orange')

            else:
                plt.scatter(x, y, s=0.1, color='b')
                ax = plt.gca()
                cset = ax.contour(X, Y, Z, colors='k', levels=[0.01, 0.05, 0.1, 0.5])
                ax.clabel(cset, inline=1, fontsize=10)
                if top_min:
                    ax.vlines(top_min, *ax.get_ylim(), colors='orange')
                ax.vlines(top_min, *ax.get_ylim(), colors='orange')

            ax.legend(["Comparison data", f"Top {top_percent * 100}% divide "])
            ax.set_xlabel("Similarity ({} fingerprint / \n{} similarity score)".format(*x_name.split("_")))
            ax.set_ylabel(
                "{}Difference in \n{} values".format("Absolute " if prop_abs else "", y_name.split("_")[1].upper()))
            # ax.text(x.max() - 0.13, y.max() - 1, f"Area: {percent_top_area * 100:.2f}%", fontsize=14)
            if save_fig:
                plt.savefig(os.path.join(plot_dir, f"{'3D' if plot_3d else ''}SingleKDEPlt_kde"
                                                   f"{int(anal_percent * 100):02d}_{int(top_percent * 100):02d}top_"
                                                   f"{x_name}_{y_name.strip('diff_')}{'_abs' if prop_abs else ''}.png"),
                            dpi=300)
        return (percent_top_area, kernel) if return_kernel else percent_top_area

    @staticmethod
    def ranking_ratio(data_df, top_percent=0.10, anal_percent=1, x_name="mfpReg_tanimoto", y_name="diff_homo"):
        """
        Get the percent of the top ranked similar molecules that are also the top ranked in terms of property difference
        Args:
            data_df (DataFrame): DataFrame containing the data.
            top_percent (float, optional): Percentage of top data (as decimal) to compare with the entire KDE. Default is 0.10.
            anal_percent (float, optional): Top percentage of data_df to include in the analysis. Default is 1 (all).
            x_name (str, optional): Name of the column representing the independent variable. Default is "mfpReg_tanimoto".
            y_name (str, optional): Name of the column representing the dependent variable. Default is "diff_homo".
        """
        data_df.sort_values(by=[x_name, y_name], ascending=[False, True], inplace=True)
        ranking_data = data_df[:int(len(data_df.index) * anal_percent)]

        sim_sorted = ranking_data[[x_name, y_name]].sort_values(by=[x_name, y_name], ascending=[False, True])
        prop_sorted = ranking_data[[x_name, y_name]].sort_values(by=[y_name, x_name], ascending=[True, False])
        cutoff_idx = int(len(prop_sorted.index) * top_percent)
        prop_top_cutoff = prop_sorted.iloc[cutoff_idx][y_name]
        top_sim_props = sim_sorted[:cutoff_idx][y_name]
        ranking_ratio = (top_sim_props < prop_top_cutoff).sum() / top_sim_props.count()

        return ranking_ratio

    def _get_total_docs(self):
        raise NotImplementedError

    def _random_sample(self, size=1000, **kwargs):
        raise NotImplementedError

    def random_kde(self, x=None, y=None, size=1000, rand_seed=1, return_df=False, **kwargs):
        """
        Samples a random subset of documents from a MongoDB collection, performs KDE analysis, and returns
        the top percentile area.

        Parameters:
        x (str, optional): The field to be used as the x-axis for KDE analysis. Default is self.default_sim.
        y (str, optional): The field to be used as the y-axis for KDE analysis. Default is self.default_prop.
        size (int, optional): Number of documents to sample. Default is 1000.
        return_df (bool, optional): If True, returns both the KDE top percentile area and the sampled DataFrame.

        Returns:
        float or tuple: The KDE top percentile area. If `return_df` is True, returns a tuple (KDE top percentile area).
        """
        x = x or self.default_sim
        y = y or self.default_sim
        sample_pairs_csv = self.data_dir / "random_sampling" / f"Combo_{size}size_{rand_seed:02d}.csv"
        if not os.path.isfile(sample_pairs_csv):
            df = self._random_sample(size=size)
        else:
            df = pd.read_csv(sample_pairs_csv, index_col=0)
        print("Starting analysis...") if self.verbose else None
        perc = self.kde_integrals(df, x_name=x, y_name=y, anal_percent=self.anal_percent, plot_dir=self.plot_dir,
                                  verbose=self.verbose, **kwargs)
        return (perc, df) if return_df else perc

    def _generate_all_kde_df(self, sample_pairs_df, **kwargs):
        """
        Generates a DataFrame containing the Kernel Density Estimation (KDE) integrals for specified similarity metrics and properties.

        Parameters:
        sample_pairs_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.
        **kwargs: Additional keyword arguments to pass to the `kde_integrals` function.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """
        # Initialize DataFrame to store KDE integral results
        area_df = pd.DataFrame(index=range(len(self.sim_cols)), columns=[])
        area_df["sim"] = self.sim_cols
        sample_pairs_df.fillna(0, inplace=True)
        print("--> Starting KDE integral analysis.") if self.verbose > 0 else None
        for prop in self.prop_cols:
            area_df[prop] = area_df.swifter.apply(
                lambda x: self.kde_integrals(sample_pairs_df, anal_percent=self.anal_percent,
                                             top_percent=self.top_percent, verbose=self.verbose,
                                             x_name=x.sim, y_name=prop, plot_dir=self.plot_dir, **kwargs),
                axis=1)
            print("--> Finished KDE integral analysis for {}.".format(prop)) if self.verbose > 1 else None
        area_df.set_index("sim", inplace=True)
        if "diff_hl" in area_df.columns:
            area_df.sort_values("diff_hl", inplace=True)
        
        return area_df

    def _generate_all_nhr_df(self, sample_pairs_df, **kwargs):
        """
        Parameters:
        sample_pairs_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """
        # Initialize DataFrame to store KDE integral results
        area_df = pd.DataFrame(index=range(len(self.sim_cols)), columns=[])
        area_df["sim"] = self.sim_cols
        sample_pairs_df.fillna(0, inplace=True)
        print("--> Starting NHR integral analysis.") if self.verbose > 0 else None
        from similarities.neighborhood_ratios import enhancement_ratio
        for prop in [self.default_prop]:
            area_df[prop] = area_df.swifter.apply(
                lambda x: enhancement_ratio(sample_pairs_df, x_name=x.sim, y_name=prop, **kwargs), axis=1)
            print("--> Finished NHR integral analysis for {}.".format(prop)) if self.verbose > 1 else None
        area_df.set_index("sim", inplace=True)
        if "diff_hl" in area_df.columns:
            area_df.sort_values("diff_hl", inplace=True)

        return area_df

    def _generate_all_ranking_df(self, sample_pairs_df):
        """
        Parameters:
        sample_pairs_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """
        # Initialize DataFrame to store KDE integral results
        area_df = pd.DataFrame(index=range(len(self.sim_cols)), columns=[])
        area_df["sim"] = self.sim_cols
        sample_pairs_df.fillna(0, inplace=True)
        print("--> Starting ranking integral analysis.") if self.verbose > 0 else None
        for prop in [self.default_prop]:
            area_df[prop] = area_df.swifter.apply(
                lambda x: self.ranking_ratio(sample_pairs_df, x_name=x.sim, y_name=prop, anal_percent=self.anal_percent, top_percent=self.top_percent), axis=1)
            print("--> Finished Ranking integral analysis for {}.".format(prop)) if self.verbose > 1 else None
        area_df.set_index("sim", inplace=True)
        if "diff_hl" in area_df.columns:
            area_df.sort_values("diff_hl", inplace=True)

        return area_df

    def _random_similarities(self, base_df, ):
        """
        Parameters:
        base_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """
        num_rows = base_df.shape[0]
        new_df = base_df.copy()

        for p in self.prop_cols:
            new_df[p] = np.random.uniform(0, 1, num_rows)

        for s in [s for s in self.sim_cols if s in new_df.columns]:
            new_df[s] = np.random.uniform(0, 1, num_rows)
        return new_df

    def _uniform_similarities(self, base_df, corr_cutoff=False, prop="diff_homo"):
        """
        Parameters:
        base_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """
        num_rows = base_df.shape[0]
        new_df = base_df.copy()

        for p in self.prop_cols:
            new_df[p] = new_df[p].abs()

        for s in self.sim_cols:
            if s in new_df.columns:
                new_df[s] = np.random.uniform(0, 1, num_rows)

            if corr_cutoff:
                # Replace similarity columns with new random data until all values are less than _temp
                new_df["corr_line"] = MinMaxScaler().fit_transform(np.array(-new_df[prop]).reshape(-1, 1))
                mask = new_df[s] >= new_df["corr_line"]
                while mask.sum() >= 2:
                    new_values = np.random.uniform(0, 1, mask.sum())
                    new_df.loc[mask, s] = new_values
                    mask = new_df[s] >= new_df["corr_line"]

        return new_df

    def _correlated_similarities(self, base_df, prop="diff_homo"):
        """
        Parameters:
        base_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """
        pd.options.mode.copy_on_write = True
        num_rows = base_df.shape[0]
        new_df = base_df.copy()
        for sim in self.sim_cols:
            new_df[sim] = MinMaxScaler().fit_transform(np.array(-new_df[prop].abs()).reshape(-1, 1))
            new_df[sim] = new_df[sim] + np.random.normal(0, 0.01, num_rows)

        return new_df

    def _normal_similarities(self, base_df, prop="diff_homo", std_dev=0.5, corr_cutoff=False):
        """
        Parameters:
        base_df (pd.DataFrame): DataFrame containing the sample pairs data with similarity metrics and properties.
        prop (str, optional): The property column to use for correlation-based cutoff adjustments. Default is "diff_homo".
        std_dev (float, optional): The standard deviation for the truncated normal distribution used to generate similarity values. Default is 0.5.
        corr_cutoff (bool, optional): If True, similarity values are adjusted iteratively to ensure they remain below a dynamically calculated
            correlation threshold based on the `prop` column. Default is False.

        Returns:
        pd.DataFrame: DataFrame with similarity metrics as rows and properties as columns, containing the KDE integral values.
        """

        num_rows = base_df.shape[0]
        new_df = base_df.copy()
        for p in self.prop_cols:
            new_df[p] = new_df[p].abs()

        for s in self.sim_cols:
            # Generate truncated normally distributed numbers
            a_trunc, b_trunc, loc = 0, 1, 0
            a, b = (a_trunc - loc) / std_dev, (b_trunc - loc) / std_dev
            new_df[s] = truncnorm(a, b, loc=loc, scale=std_dev).rvs(size=num_rows)

            if corr_cutoff:
                # Replace similarity columns with new random data until all values are less than _temp
                new_df["corr_line"] = MinMaxScaler().fit_transform(np.array(-new_df[prop]).reshape(-1, 1))
                mask = new_df[s] >= new_df["corr_line"]
                while mask.sum() >= 2:
                    new_values = truncnorm(a, b, loc=loc, scale=std_dev).rvs(size=mask.sum())
                    new_df.loc[mask, s] = new_values
                    mask = new_df[s] >= new_df["corr_line"]

        return new_df

    def _get_sample_pairs_df(self, i, size, anal_percent=1, replace_sim=False, norm_std_dev=None):
        """
        Generates or retrieves a DataFrame of sampled document pairs for analysis, with optional similarity replacement.

        Parameters:
        i (int):
            The trial number used to name and retrieve saved files.
        size (int):
            The number of documents to sample for each trial.
        anal_percent (float, optional):
            The percentage of the dataset to analyze. Default is 1 (100%).
        replace_sim (str or bool, optional):
            Method for replacing similarity values. Options include "uniform", "uniformCorr", "correlated",
            "normal", "normalCorr", or "random". Default is False (no replacement).
        norm_std_dev (float, optional):
            The standard deviation for normal distribution replacement methods. Used when `replace_sim` is "normal"
            or "normalCorr".

        Returns:
        pd.DataFrame:
            A DataFrame containing the sampled similarity and property columns, with optional replacements applied.
        """
        comp_dir = self.data_dir / "random_sampling"

        # If sample_pairs_csv does not exist, generate a new random sample and save to CSV
        sample_pairs_csv = comp_dir / f"Combo_{size}size_{i:02d}.csv"
        if not os.path.isfile(sample_pairs_csv):
            # Establish one variable, _working_df, so only one DataFrame is held in memory
            _working_df = self._random_sample(size=size, anal_percent=anal_percent)
            _working_df.to_csv(sample_pairs_csv)
        else:
            _working_df = pd.read_csv(sample_pairs_csv, index_col=0)

        # Generate KDE integrals DataFrame and save to CSV
        if replace_sim:
            replace_csv = comp_dir / f"Combo_{size}size_{i:02d}_{replace_sim}.csv"
            if not os.path.isfile(replace_csv) or self.replace_files:

                # Generate replacement data
                if replace_sim == "uniform":
                    _working_df = self._uniform_similarities(_working_df)
                elif replace_sim == "uniformCorr":
                    _working_df = self._uniform_similarities(_working_df, corr_cutoff=True)
                elif replace_sim == "correlated":
                    _working_df = self._correlated_similarities(_working_df)
                elif replace_sim == "normal":
                    _working_df = self._normal_similarities(_working_df, std_dev=norm_std_dev)
                elif replace_sim == "normalCorr":
                    _working_df = self._normal_similarities(_working_df, corr_cutoff=True, std_dev=norm_std_dev)
                elif replace_sim == "random":
                    _working_df = self._random_similarities(_working_df)
                else:
                    raise Exception(f"No replacement method found for replace_sim {replace_sim}.")

                _working_df.to_csv(replace_csv)
            else:
                _working_df = pd.read_csv(replace_csv, index_col=0)

        return _working_df[self.sim_cols + self.prop_cols]

    @staticmethod
    def get_similarity_measures_below_upper_bound(avg_df: pd.DataFrame) -> list:
        """
        Returns a list of similarity measure names where the standard deviation lower bound
        (mean - std) is less than the standard deviation upper bound (mean + std) of the first similarity measure.

        Parameters:
        avg_df (pd.DataFrame): DataFrame containing the average values.

        Returns:
        list: A list of similarity measure names that meet the condition.
        """
        # Calculate mean and standard deviation across rows
        mean_row = avg_df.mean(axis=1)
        std_row = avg_df.std(axis=1)

        # Similarity measure names where the lower bound is less than the upper bound of the first similarity measure
        std_upper_bound_first = mean_row.iloc[0] + std_row.iloc[0]
        lower_bounds = mean_row - std_row
        similarity_measures = lower_bounds[lower_bounds < std_upper_bound_first].index.tolist()

        return similarity_measures

    def plot_avg_df(self, avg_df, ylims=None, ax=None, std_values=None, return_plot=True, red_labels=True, 
                    upper_bound=None, lower_bound=None, soft_upper_bound=None, soft_lower_bound=None,
                    ratio_name=None, anal_name=None, main_plot_color = "blue"):
        """
        Plots average values from a DataFrame with optional standard deviation shading, bounds, and customized annotations.

        Parameters:
        avg_df (pd.DataFrame):
            DataFrame containing average values to plot, indexed by similarity measures.
        ylims (tuple, optional):
            Tuple specifying the y-axis limits (lower, upper). Default is None.
        ax (matplotlib.axes.Axes, optional):
            Existing matplotlib Axes object to plot on. Default is None (creates a new Axes).
        std_values (pd.DataFrame, optional):
            DataFrame of standard deviation values for additional plotting. Default is None.
        return_plot (bool, optional):
            If True, returns the Axes object. Default is True.
        red_labels (bool, optional):
            If True, highlights similarity measures below the upper bound in red. Default is True.
        upper_bound (float, optional):
            Horizontal line indicating the upper bound. Default is None.
        lower_bound (float, optional):
            Horizontal line indicating the lower bound. Default is None.
        soft_upper_bound (float, optional):
            Horizontal line indicating a weak upper bound. Default is None.
        soft_lower_bound (float, optional):
            Horizontal line indicating a weak lower bound. Default is None.
        ratio_name (str, optional):
            Label for the y-axis to denote the metric being analyzed. Default is None.
        anal_name (str, optional):
            Title of the plot, typically describing the analysis. Default is None.
        main_plot_color (str, optional):
            Color for the primary plot elements. Default is "blue".

        Returns:
        matplotlib.axes.Axes or pd.DataFrame:
            The plot's Axes object if `return_plot` is True, otherwise the modified DataFrame.
        """

        if std_values is not None:
            
            sort_value = (std_values.mean(axis=1) + std_values.std(axis=1)).sort_values()
            avg_df = avg_df.reindex(sort_value.index)
            std_values = std_values.reindex(list(sort_value.index))
        
            ax = sns.scatterplot(avg_df, s=10)
            std_mean, std_stdev = std_values.mean(axis=1), std_values.std(axis=1)
            ax.plot(std_mean, label='Mean', color='blue')
            # ax.fill_between(std_mean.index, std_mean - std_stdev, std_mean + std_stdev, color='blue', alpha=0.2,
            #                 label='Full Dataset Values')
            main_plot_color = "red"
        else:
            ax = sns.scatterplot(avg_df, s=10, ax=ax)
        mean_row, std_row = avg_df.mean(axis=1), avg_df.std(axis=1)
        ax.plot(mean_row, label='Mean', color=main_plot_color)
        ax.fill_between(mean_row.index, mean_row - std_row, mean_row + std_row, color=main_plot_color, alpha=0.2,
                        label='1 Std Dev')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        # Set x lables
        ax.set_xticks(range(0, len(avg_df.index)), avg_df.index, rotation="vertical", fontsize=10)
        equivilant_sims = self.get_similarity_measures_below_upper_bound(avg_df)
        if red_labels: 
            for label in ax.get_xticklabels():
                if label.get_text() in equivilant_sims:
                    label.set_color('red')

        if ylims:
            ax.set_ylim(*ylims)
        if upper_bound is not None:
            ax.axhline(y=upper_bound, color='black', linestyle='dashdot', linewidth=2, label=f'Upper Bound ({upper_bound})')
        if soft_upper_bound is not None:
            ax.axhline(y=soft_upper_bound, color='black', linestyle='dotted', linewidth=2, label=f'Weak Upper Bound ({soft_upper_bound})')

        if soft_lower_bound is not None:
            ax.axhline(y=soft_lower_bound, color='black', linestyle='dashed', linewidth=2, label=f'Weak Lower Bound ({soft_lower_bound})')
        if lower_bound is not None:
            ax.axhline(y=lower_bound, color='black', linestyle='solid', linewidth=2, label=f'Lower Bound ({lower_bound})')

        ax.set_xlabel("Similarity Measure")
        y_axis_name = "Top Area Ratio" if ratio_name == "IntegralRatios" else ratio_name
        ax.set_ylabel(f"Average {y_axis_name}")
        ax.set_title(anal_name.replace("_", " ").capitalize())
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"Avg{ratio_name}_{anal_name}.png", dpi=300)
        print("Done. Plots saved") if self.verbose else None
        if return_plot:
            return ax
        return avg_df

    def random_sampling(self, size, num_trials=30, plot=True, replace_sim=None, method="kde", norm_std_dev=None, t_x=None,
                       **plotting_kwargs):
        """
        Performs random sampling analysis by sampling multiple datasets, applying analysis methods (e.g., KDE),
        and aggregating the results.

        Parameters:
        size (int): Number of documents to sample for each trial.
        num_trials (int, optional): Number of trials to perform. Default is 30.
        plot (bool, optional): If True, plots the aggregated results. Default is True.
        replace_sim (str, optional): Replacement strategy for similarity measures (e.g., "uniform", "normal"). Default is None.
        method (str, optional): Analysis method to use ("kde", "nhr", or "ranking"). Default is "kde".
        norm_std_dev (float, optional): Standard deviation for normal similarity replacement. Default is None.
        t_x (float, optional): Intercept parameter for neighborhood ratios. Default is None.
        **plotting_kwargs: Additional keyword arguments for plotting.

        Returns:
        pd.DataFrame or matplotlib.axes.Axes:
            Aggregated DataFrame of average values for each trial. If `plot=True`, returns the Axes object.
        """

        avg_dfs = []
        comp_dir = self.data_dir / "random_sampling"
        ratio_name = "IntegralRatios" if method=="kde" else "NeighborhoodRatios" if method=="nhr" else "RankingRatios" if method=="ranking" else ""

        # Iterate through multiple trials
        anal_name = f"{size}size_{self.perc_name}" + (f"_{replace_sim}" if replace_sim else "") + (f"_incpt{t_x}" if t_x is not None else "")
        for i in range(num_trials):
            print("Creating data sample with random seed {}...".format(i)) if self.verbose else None

            # Check if the area_df_csv file exists
            area_df_csv = comp_dir / f"{ratio_name}_{anal_name}_Rand{i:02d}.csv"
            if not os.path.isfile(area_df_csv) or self.replace_files:
                _pairs_df = self._get_sample_pairs_df(i=i, size=size, replace_sim=replace_sim, norm_std_dev=norm_std_dev)
                if method == "kde":
                    _results_df = self._generate_all_kde_df(_pairs_df)
                elif method == "nhr":
                    _results_df = self._generate_all_nhr_df(_pairs_df, t_x=t_x)
                elif method == "ranking":
                    _results_df = self._generate_all_ranking_df(_pairs_df)
                else:
                    raise KeyError(f"Method {method} is not an available method." )
                _results_df.to_csv(area_df_csv)
            else:
                _results_df = pd.read_csv(area_df_csv, index_col=0)

            # Append the average series of the DataFrame to avg_dfs list
            avg_dfs.append(pd.Series(_results_df.mean(axis=1)))

        print("Concatenate all average series into avg_df DataFrame...") if self.verbose > 1 else None
        avg_df = pd.concat(avg_dfs, axis=1)

        # Sort avg_df by the maximum value in each row
        sort_value = (avg_df.mean(axis=1) + avg_df.std(axis=1)).sort_values(ascending=True if method=="kde" else False)
        avg_df = avg_df.reindex(sort_value.index)

        # Save avg_df to CSV
        print(f"Saving data to file: Avg{ratio_name}_{anal_name}.csv...", ) if self.verbose > 1 else None
        avg_df.to_csv(comp_dir / f"Avg{ratio_name}_{anal_name}.csv")

        # Plotting if plot=True
        if plot:
            print("Plotting data...") if self.verbose > 1 else None
            return self.plot_avg_df(avg_df, ratio_name=ratio_name, anal_name=anal_name+f"{num_trials:02d}trials", **plotting_kwargs)
        return avg_df

    def random_sampling_percentiles(self, size, num_trials=30, plot=True, replace_sim=None, ylims=None, ax=None,
                                   std_percentiles=None):
        """
        Performs random sampling analysis to calculate percentile divides for similarity measures and optionally plots the results.

        Parameters:
        size (int): Number of documents to sample for each trial.
        num_trials (int, optional): Number of trials to perform. Default is 30.
        plot (bool, optional): If True, plots the percentile divides. Default is True.
        replace_sim (str, optional): Replacement strategy for similarity measures (e.g., "uniform", "normal"). Default is None.
        ylims (tuple, optional): Y-axis limits for the plot. Default is None.
        ax (matplotlib.axes.Axes, optional): Axes object for plotting. If None, a new plot is created. Default is None.
        std_percentiles (pd.Series, optional): Pre-calculated percentiles to overlay on the plot. Default is None.

        Returns:
        pd.DataFrame or tuple:
            DataFrame of calculated percentile divides for each similarity measure.
            If `plot=True`, returns a tuple (DataFrame, Axes object).
        """

        divides_df = pd.DataFrame(index=self.sim_cols, columns=[i for i in range(num_trials)])
        comp_dir = self.data_dir / "random_sampling"

        # Iterate through multiple trials
        anal_name = f"{size}size_{self.perc_name}" + (replace_sim if replace_sim else "")
        for i in range(num_trials):
            working_df = self._get_sample_pairs_df(i=i, size=size, replace_sim=replace_sim)
            for sim in self.sim_cols:
                divide = working_df[sim].quantile(self.percentile / 100)
                divides_df.at[sim, i] = divide
                print(
                    "f{self.percentile} percentile divide calculated for {sim}: {divide}") if self.verbose > 1 else None

        # Sort divides_df by the maximum value in each row
        sort_value = (divides_df.mean(axis=1) + divides_df.std(axis=1)).sort_values()
        divides_df = divides_df.reindex(sort_value.index)

        # Plotting if plot=True
        if plot:
            mean_row, std_row = divides_df.mean(axis=1), divides_df.std(axis=1)
            ax = sns.scatterplot(divides_df, s=10, ax=ax)
            if std_percentiles is not None:
                sorted_percentiles = std_percentiles.reindex(list(divides_df.index))
                ax.plot(sorted_percentiles)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.set_xticks(range(0, len(divides_df.index)), divides_df.index, rotation="vertical", fontsize=10)
            if ylims:
                ax.set_ylim(*ylims)
            ax.set_xlabel("Similarity Measure")
            ax.set_ylabel(f"{self.percentile} Percentile Divides")
            ax.set_title(anal_name.replace("_", " ").capitalize())
            plt.tight_layout()
            plt.savefig(self.plot_dir / f"Divides_{anal_name}_{num_trials:02d}trials.png", dpi=300)
            print("Done. Plots saved") if self.verbose else None
            return divides_df, ax
        return divides_df


class SimilarityAnalysisRand(SimilarityAnalysisBase):
    def __init__(self, anal_percent: float = 1, top_percent: float = 1, orig_df: pd.DataFrame = None, smiles_pickle: str = None,
                 verbose: int = 3, anal_name: str = "SimAnalysis", elec_props: list = ELEC_PROPS,
                 sim_metrics: dict = SIM_METRICS, fp_gens: dict = FP_GENS, **kwargs):
        """

        Parameters:
        anal_percent (float): Top percent of data to use in the analysis
        top_percent (float): Top percent of the data to use to calculate the top area ratio
        total_docs (int): Total number of molecule pair data entries
        verbose (int): Verbosity level (0 = silent, 1 = minimal output, 2 = detailed output).
        elec_props (list): List of electronic properties to calculate differences.
        sim_metrics (dict): Dictionary of similarity metrics to calculate.
        fp_gens (dict): Dictionary of fingerprint generation methods.
        mongo_uri (str): MongoDB connection URI.
        mongo_db (str): MongoDB database name.
        mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

        """
        self.molecules_df = self._generate_molecules_df(orig_df=orig_df, smiles_pickle=smiles_pickle, fp_dict=fp_gens)
        super().__init__(anal_percent=anal_percent, top_percent=top_percent, verbose=verbose, anal_name=anal_name,
                         elec_props=elec_props, sim_metrics=sim_metrics, fp_gens=fp_gens, **kwargs)
        self.all_ids = self.molecules_df.index.tolist()

    @staticmethod
    def _generate_molecules_df(orig_df: pd.DataFrame = None, smiles_pickle: str = None, fp_dict=FP_GENS):
        """
        Generates a DataFrame of molecules with specified fingerprints from either an original DataFrame or a pickle file
        containing SMILES strings.

        Parameters:
        orig_df (pd.DataFrame): Original DataFrame containing SMILES strings and/or molecular objects.
        smiles_pickle (str): Path to a pickle file containing a DataFrame with SMILES strings.
        fp_dict (dict): Dictionary of fingerprint generation methods.

        Returns:
        pd.DataFrame: DataFrame containing molecules with generated fingerprints.
            """
        if orig_df is None:
            if not os.path.isfile(smiles_pickle):
                raise IOError("No DF pickle file found at {}. This function requires either an original_df argument or"
                              "a valid DF pickle file location".format(smiles_pickle))
            if smiles_pickle.endswith(".csv"): 
                orig_df = pd.read_csv(smiles_pickle)
            else: 
                orig_df = pd.read_pickle(smiles_pickle)

        if 'mol' not in orig_df.columns:
            if 'smiles' not in orig_df.columns:
                raise KeyError(f'Column "smiles" not found in {smiles_pickle} columns')
            orig_df['mol'] = orig_df.smiles.apply(lambda x: Chem.MolFromSmiles(x))
        for fp_name, fpgen in fp_dict.items():
            print(f"FP Generation Method: {fp_name}")
            if fp_name not in orig_df.columns:
                orig_df[fp_name] = orig_df.mol.apply(lambda x: fpgen(x).ToBase64())
            else:
                orig_df[fp_name] = orig_df[fp_name].apply(lambda x: x.ToBase64())
        orig_df.index = orig_df.index.astype(str)
        return orig_df

    def _get_total_docs(self):
        """
        Calculates the total number of molecular pairs using the combination formula.

        Returns:
        int: Total number of molecular pairs.
        """
        num_mols = self.molecules_df.shape[0]
        return comb(num_mols, 2)

    def _rand_id(self):
        """
        Generates a random unique pair identifier for molecule IDs.

        Returns:
        str: Pair identifier in the format 'id1__id2', where id1 and id2 are sorted.
        """
        id_1 = id_2 = random.choice(self.all_ids)
        while id_1 == id_2:
            id_2 = random.choice(self.all_ids)
        return "__".join(sorted([str(id_1), str(id_2)]))

    def _get_pair_data(self, pair_id, elec_props=None, fp_dict=None, sim_metrics=None, **kwargs):
        """
        Generates data for a given pair of molecules by calculating property differences and similarity metrics.

        Parameters:
        pair_id (str): Unique identifier for the molecule pair in the format 'id1__id2'.
        elec_props (list, optional): List of electronic properties to compute differences for. Default is None.
        fp_dict (dict, optional): Dictionary of fingerprints for each molecule. Default is None.
        sim_metrics (dict, optional): Dictionary of similarity metrics and their calculation methods. Default is None.

        Returns:
        dict: Dictionary containing pair data, including property differences and similarity scores.
        """
        id_1, id_2 = pair_id.split("__")
        id_1_dict = self.molecules_df.loc[id_1].to_dict()
        id_2_dict = self.molecules_df.loc[id_2].to_dict()

        insert_data = {"_id": pair_id, "id_1": id_1, "id_2": id_2}
        # Add electronic property differences
        for ep in (elec_props or self.elec_props):
            insert_data["diff_" + ep] = id_1_dict[ep] - id_2_dict[ep]

        # Add similarity metrics
        for fp in (fp_dict or self.fp_dict).keys():
            print(f"FP Generation Method: {fp}") if self.verbose > 3 else None
            for sim, SimCalc in (sim_metrics or self.sim_metrics).items():
                print(f"\tSimilarity {sim}") if self.verbose > 3 else None
                metric_name = f"{fp}_{sim.lower()}"
                fp1 = ExplicitBitVect(2048)
                fp1.FromBase64(id_1_dict[fp])
                fp2 = ExplicitBitVect(2048)
                fp2.FromBase64(id_2_dict[fp])
                similarity = SimCalc(fp1, fp2)
                insert_data[f"{metric_name}"] = similarity
        return insert_data

    def _random_sample(self, size=1000, **kwargs):
        """
        Generates a random sample of molecule pairs and their associated data.

        Parameters:
        size (int, optional): Number of random pairs to generate. Default is 1000.

        Returns:
        pd.DataFrame: DataFrame containing data for the sampled pairs, indexed by pair ID.
        """
        ids = set([self._rand_id() for _ in range(size)])
        while len(ids) < size:
            ids = set(list(ids) + [self._rand_id()])

        print(f"Generating random sample") if self.verbose > 3 else None
        all_data = [self._get_pair_data(i, **kwargs) for i in ids]

        sample_df = pd.DataFrame(all_data)
        sample_df.set_index("_id", inplace=True)
        return sample_df


class SimilarityPairsDBAnalysis(SimilarityAnalysisBase):
    def __init__(self, anal_percent = 1, top_percent = 1,
                 total_docs=353314653, verbose=3,
                 elec_props=ELEC_PROPS, sim_metrics=SIM_METRICS, fp_gens=FP_GENS,
                 mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL, **kwargs):
        """

        Parameters:
        anal_percent (float): Top percent of data to use in the analysis
        top_percent (float): Top percent of the data to use to calculate the top area ratio
        total_docs (int): Total number of molecule pair data entries
        verbose (int): Verbosity level (0 = silent, 1 = minimal output, 2 = detailed output).
        elec_props (list): List of electronic properties to calculate differences.
        sim_metrics (dict): Dictionary of similarity metrics to calculate.
        fp_gens (dict): Dictionary of fingerprint generation methods.
        mongo_uri (str): MongoDB connection URI.
        mongo_db (str): MongoDB database name.
        mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

        """
        self.total_docs = total_docs
        super().__init__(anal_percent=anal_percent, top_percent=top_percent, verbose=verbose, anal_name=mongo_coll,
                         elec_props=elec_props, sim_metrics=sim_metrics, fp_gens=fp_gens, **kwargs)
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_coll = mongo_coll

    def _get_total_docs(self):
        """
        Fetches the total number of documents from a MongoDB collection.

        Returns:
        int: Total number of documents in the specified collection.
        """
        if self.total_docs:
            return self.total_docs
        print("Starting total number of docs query...") if self.verbose else None
        with MongoClient(self.mongo_uri) as client:
            return client[self.mongo_db][self.mongo_coll].count_documents({})

    def find_percentile(self, field, percentile):
        """
        Finds the value at a specified percentile for a given field in a MongoDB collection.

        Parameters:
        field (str): The field for which the percentile value is to be calculated.
        percentile (float): The percentile to find (between 0 and 100).

        Returns:
        float: The value at the specified percentile for the given field.
        """
        percentile = percentile or self.percentile
        if percentile < 50:
            sort_dir = pymongo.ASCENDING
            percentile_idx = int(self.total_docs * (percentile / 100))
        else:
            sort_dir = pymongo.DESCENDING
            percentile_idx = int(self.total_docs * ((100 - percentile) / 100))
        print(f"Percentile index {percentile_idx} out of total {self.total_docs} documents.") if self.verbose else None

        print("Starting sort and skip query..") if self.verbose > 2 else None
        with MongoClient(self.mongo_uri) as client:
            doc = client[self.mongo_db][self.mongo_coll].find_one({}, {field: 1}, sort=[(field, sort_dir)],
                                                                  skip=percentile_idx,
                                                                  allow_disk_use=True)

            return doc[field]

    def _random_sample(self, x=None, y=None, anal_percent=None, size=1000):
        """
        Retrieves a random sample of documents from a MongoDB collection, with options to sort, limit,
        and project specific fields.

        Parameters:
        x (str, optional): Field to sort by in descending order. Default is None.
        y (str, optional): Additional field to include in the projection. Default is None.
        anal_percent (float, optional): Top percent of data to use in the analysis. Default is None.
        size (int, optional): Number of documents to sample. Default is 1000.

        Returns:
        pd.DataFrame: DataFrame containing the sampled documents, indexed by MongoDB's `_id` field.
        """
        # Create the aggregation pipeline for MongoDB
        pipeline = [{"$sample": {"size": size}}]  # Randomly select 'size' number of documents
        pipeline.append(
            {"$limit": size * anal_percent}) if anal_percent else None  # Limit to the top 'num_top_docs' documents
        pipeline.append({"$sort": {x: -1}}) if x else None  # Sort by the specified field in ascending order
        pipeline.append({'$project': {v: 1 for v in [x, y] if v}}) if (
                x or y) else None  # Include only the fields 'x' and 'y'

        with MongoClient(self.mongo_uri) as client:
            print(
                f"Starting query to select top {size * anal_percent} of {size} random docs...") if self.verbose else None
            results = list(client[self.mongo_db][self.mongo_coll].aggregate(pipeline, allowDiskUse=True))
            df = pd.DataFrame(results).set_index("_id")
        return df

    def _batch_kde(self, sim=None, prop=None, batch_size=10000, zeros_cutoff=1e-10, divide=None,
                   **kwargs):
        """
        Performs batch Kernel Density Estimation (KDE) analysis on a dataset stored in a MongoDB collection
        and returns the percent of the top integrated KDE area.

        Parameters:
        sim (str): Similarity metric to use for KDE analysis. Default is self.default_sim.
        prop (str): Property to analyze alongside the similarity metric. Default is self.default_prop.
        batch_size (int): Number of documents to process in each batch. Default is 10,000.
        zeros_cutoff (float): Threshold below which the KDE percent is considered zero. Default is 1e-10.
        divide (float, optional): Minimum value for KDE top area. Default is None.

        Returns:
        float: Percent of the KDE top area.
        (pd.DataFrame, float): If return_all_d is True, returns a tuple with the average percent and the results.
        """

        # Set up batch analysis
        sim = sim or self.default_sim
        prop = prop or self.default_prop
        anal_num = round(self.total_docs * self.anal_percent)
        total_processed = 0
        result_list = []
        perc = 1

        print(f"Starting analysis for {anal_num} docs...") if self.verbose > 1 else None
        b_dfs = []
        while total_processed < anal_num:
            # Fetch the batch of documents
            current_batch_size = min(batch_size, anal_num - total_processed)
            sort_dir = -1 if (total_processed < self.total_docs / 2) else 1
            skip_num = total_processed if (total_processed < self.total_docs / 2) else anal_num - (
                    total_processed - current_batch_size)

            if perc < zeros_cutoff:  # The percents keep getting lower. So, if the previous percent < divide, perc ~ 0
                perc, kernel = 0, None
            else:
                print(f"...Starting query for {current_batch_size}: sorting {sort_dir} and skipping "
                      f"{skip_num}...") if self.verbose > 2 else None
                with MongoClient(self.mongo_uri) as client:
                    cursor = client[self.mongo_db][self.mongo_coll].find({}, {sim: 1, prop: 1},
                                                                         allow_disk_use=True).sort(
                        {sim: sort_dir, prop: 1, "_id": 1}).skip(skip_num).limit(current_batch_size)
                    b_df = pd.DataFrame(list(cursor))
                    b_dfs.append(b_df)

                # Apply the analysis function to the DataFrame
                d_min, d_max = b_df[sim].min(), b_df[sim].max()
                print(f"  --> DF min {d_min}, DF max {d_max}") if self.verbose > 2 else None

                if d_min == d_max:  # If the DF covers no range, the percent KDE will fail.
                    perc, kernel = 1 if d_min > divide else 0, None
                else:
                    perc, kernel = self.kde_integrals(b_df, x_name=sim, y_name=prop, anal_percent=1, top_min=divide,
                                                      top_percent=self.top_percent, verbose=self.verbose,
                                                      plot_dir=self.plot_dir, return_kernel=True, **kwargs)
            result_list.append({"percent": perc, "length": b_df.shape[0], "kernel": kernel})
            print(f"  --> Processed {total_processed} documents. Current KDE analysis ({current_batch_size} docs) "
                  f"yields {perc}%") if self.verbose > 1 else None

            # Update the counter
            processed_count = b_df.shape[0]
            total_processed += processed_count

        results = pd.DataFrame(result_list)
        results.to_csv(self.data_dir / f"BatchKDE_{sim}_{prop}_{self.perc_name}.csv")
        avg_perc = (results["percent"] * results["length"]).sum() / results["length"].sum()
        print(f"KDE Top Area for {sim} and {prop}: {avg_perc}") if self.verbose else None

        return avg_perc, results

    def db_kde(self, sim=None, prop=None, divide=None, batch_size=None,
               return_kernel=False, replace=False, set_in_progress=False, **kwargs):
        """
        Performs Kernel Density Estimation (KDE) for a specific similarity metric and property,
        using either batch processing or the full dataset, and saves the result.

        Parameters:
        sim (str): Similarity metric to use for KDE analysis. Default is self.default_sim.
        prop (str): Property to analyze alongside the similarity metric. Default is self.default_prop.
        divide (float, optional): Minimum value for KDE top area. Default is None.
        batch_size (int, optional): Batch size for processing documents. Default is None.
        return_kernel (bool): Whether to return kernel results. Default is False.
        replace (bool): Whether to replace existing results. Default is False.
        set_in_progress (bool): Whether to mark the calculation as in progress. Default is False.

        Returns:
        float: The KDE top area percentage.
        (pd.DataFrame, float): If return_kernel is True, returns a tuple with the average percent and the results.
        """
        sim = sim or self.default_sim
        prop = prop or self.default_prop

        # Check of already exists
        batch_df = pd.read_csv(self.batch_kde_file, index_col=0) if os.path.isfile(
            self.batch_kde_file) else pd.DataFrame(index=self.sim_cols, columns=self.prop_cols)
        if not replace and pd.notna(batch_df.at[sim, prop]):
            print(f"--> KDE Top Area for {sim} and {prop} exists: {batch_df.at[sim, prop]}") if self.verbose else None
            return (batch_df.at[sim, prop], None) if return_kernel else batch_df.at[sim, prop]

        # Set this divide calculation in progress
        if set_in_progress:
            batch_df.at[sim, prop] = "in_progress"
            batch_df.to_csv(self.batch_kde_file)

        # Calculate the index to skip to reach the top percentile
        divide = divide or self.find_percentile(sim, percentile=self.percentile)
        print(f"{sim} {self.percentile} percentile value ({self.top_percent * 100} percentile of top "
              f"{self.anal_percent * 100}%): ", divide) if self.verbose > 1 else None

        # Set up batch analysis
        anal_num = round(self.total_docs * self.anal_percent)

        print(f"Starting analysis for {anal_num} docs...") if self.verbose > 1 else None
        if batch_size:
            perc, kernel_results = self._batch_kde(sim=sim, prop=prop, batch_size=batch_size, divide=divide, **kwargs)
        else:
            with MongoClient(self.mongo_uri) as client:
                cursor = client[self.mongo_db][self.mongo_coll].find({}, {sim: 1, prop: 1},
                                                                     allow_disk_use=True).sort(
                    {sim: -1, prop: 1, "_id": 1}).limit(anal_num)
                b_df = pd.DataFrame(list(cursor))

            perc, kernel_results = self.kde_integrals(b_df, x_name=sim, y_name=prop, anal_percent=1, top_min=divide,
                                                      top_percent=self.top_percent, verbose=self.verbose,
                                                      plot_dir=self.plot_dir, return_kernel=True, **kwargs)
        print(f"KDE Top Area for {sim} and {prop}: {perc}") if self.verbose else None

        # Save batch KDE top percent
        batch_df.at[sim, prop] = perc
        batch_df.to_csv(self.batch_kde_file)
        return (perc, kernel_results) if return_kernel else perc

    def kde_all_prop(self, sim, **kwargs):
        """
        Performs KDE analysis for all properties using the specified similarity metric.

        Parameters:
        sim (str): Similarity metric to use for KDE analysis.
        kwargs: Additional keyword arguments passed to the db_kde function.

        Returns:
        pd.DataFrame: DataFrame containing results for all properties.
        """
        for y in self.prop_cols:
            avg_perc = self.db_kde(sim=sim, prop=y, divide=self.get_divide(sim), **kwargs)
            print(f"--> KDE Top Area for {sim} and {y}: {avg_perc}") if self.verbose else None

        return pd.read_csv(self.batch_kde_file)

    def kde_all(self, **kwargs):
        """
        Performs KDE analysis for all properties and similarity measures.

        Parameters:
        kwargs: Additional keyword arguments passed to the db_kde function.

        Returns:
        pd.DataFrame: DataFrame containing results for all properties.
        """
        for sim in self.sim_cols:
            self.kde_all_prop(sim=sim, **kwargs)
        return pd.read_csv(self.batch_kde_file)

    def get_divide(self, sim_metric, replace=False, set_in_progress=False):
        """
        Fetches or calculates the divide value for a specific similarity metric.

        Parameters:
        sim_metric (str): Similarity metric for which to fetch or calculate the divide value.
        replace (bool): Whether to replace existing values. Default is False.
        set_in_progress (bool): Whether to mark the calculation as in progress. Default is False.

        Returns:
        float: The divide value.
        """

        # Establish area DF
        area_df = pd.read_csv(self.divides_file, index_col=0) if os.path.isfile(self.divides_file) else pd.DataFrame(
            index=self.sim_cols, columns=["divide"])

        # Get divide value
        if not replace and pd.notna(area_df.at[sim_metric, 'divide']):
            print(f"--> {sim_metric} {self.percentile}% value exists: {area_df.at[sim_metric, 'divide']}"
                  ) if self.verbose > 1 else None
            return area_df.at[sim_metric, 'divide']

        # Set this divide calculation in progress
        if set_in_progress:
            area_df.at[sim_metric, "divide"] = "in_progress"
            area_df.to_csv(self.divides_file)

        # Calculate divide
        divide = self.find_percentile(sim_metric, percentile=self.percentile)
        print(f"--> {sim_metric} {self.percentile} percentile value ({self.top_percent * 100} percentile of top "
              f"{self.anal_percent * 100}%): ", divide) if self.verbose > 1 else None
        area_df.at[sim_metric, "divide"] = divide
        area_df.to_csv(self.divides_file)
        return divide

    def gen_all_divides(self, replace=False):
        """
        Generates divide values for all similarity metrics.

        Parameters:
        replace (bool): Whether to replace existing values. Default is False.

        Returns:
        pd.DataFrame: DataFrame containing the divide values for all similarity metrics.
        """
        for x in self.sim_cols:
            self.get_divide(x, replace=replace)
        return pd.read_csv(self.divides_file)
