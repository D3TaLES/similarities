<p align="center">
  <img width="379" alt="image" src="https://github.com/user-attachments/assets/8039650e-26e4-4009-9e46-7c9a49bfdf8a">
</p>


KDE analysis method for evaluating molecular fingerprinting methods and similarity functions with electronic structure data. 
**Modules**: 
* `kde_analysis`: Code to use the KDE analysis for evaluating similarity measures. 
* `neighborhood_ratios`: Code to use the neighborhood ratios analysis found in [Patterson et. al.](https://doi.org/10.1021/jm960290n).
* `generate_db`: Code to generate a MongoDB database with molecule pair information.
* `settings.py`: This files contains settings for the rest of the code.

# Installation 

Clone this GitHub repository. 
```bash
git clone git@github.com:D3TaLES/roboticsUI.git
```
Create a Conda environment. 
```bash
conda create --name similarities --file similarities.yml
conda activate similarities
```

## Settings
The file `settings.py` contains several default variable that are used throughout the rest of 
the code in this repository. Before running any code, be sure the check the variables in this file. 
In particular, check the default save directories. If using a MongoDB approach, check the MongoDB connection
variables. 


# Quickstart
First, install this git repository and configure your `settings.py` file. 
Then you may begin. Start by loading your data as a pandas DataFrame before creating 
an analysis object and performing the analysis. Note that the DataFrame should contain
columns with the name of each electronic property defined in the `settings.py` file as well as a `smiles` column. 

This analysis will randomly pull `num_trials` samples of size `size` from your data and perform the KDE analysis. 
```python
import pandas as pd 
from similarities.kde_analysis import SimilarityAnalysisRand

# Load data
all_d = pd.read_pickle("path_to_dataset.pkl")

# Create analysis object
sim_anal = SimilarityAnalysisRand(anal_percent=0.20, top_percent=0.1, orig_df=all_d, anal_name="TEST")

# Perform analysis 
kde_df = sim_anal.random_sampling(size=100000, num_trials=10, plot=True, return_plot=False)
```

# Documentation 
## KDE Analysis 
### Using a local data file and performing random sampling
**When to Use**: This method is used to perform the analysis on random samples from a dataset.  
This will be the best option for most use cases. 

First, you must load your dataset as a pandas DataFrame. Note that the DataFrame should contain
columns with the name of each electronic property defined in the `settings.py` file as well as a `smiles` column.
```python
import pandas as pd 

all_d = pd.read_pickle("path_to_dataset.pkl")
```

Next, create an analysis object and perform the analysis. 
```python
from similarities.kde_analysis import SimilarityAnalysisRand

# Create analysis object
sim_anal = SimilarityAnalysisRand(anal_percent=0.10, top_percent=0.1, verbose=0, orig_df=all_d, anal_name="RandTEST")

# Perform analysis 
kde_df = sim_anal.random_sampling(size=100000, num_trials=10, plot=True, return_plot=False, upper_bound=0.172)
```

Note that the analysis instructions included the key word argument `upper_bound`. This upper bound can be found as 
described in the next section. 

#### Boundaries 
Boundaries can be easily calculated using the analysis object. The `random_sampling` function is called
but the `replace_sim` keyword argument is used to replace all similarity values with random values from some
distribution. This distribution can be either `"uniform"`, `"uniformCorr"`, `"normal"` or `"normalCorr"`.
For the most part, you should use `"uniform"` for the upper bound and `"uniformCorr"` for the lower bound.

```python
from similarities.analysis import *

sim_anal = SimilarityPairsDBAnalysis(anal_percent=0.10, top_percent=0.10, verbose=0, orig_df=all_d, anal_name="RandTEST")

upper_bound_df = sim_anal.random_sampling(size=100000, num_trials=20, plot=True, replace_sim="uniform", return_plot=False)
upper_bound = upper_bound_df.mean().mean()
```


### Using a MongoDB database
**When to Use**: This method performs analysis on an entire dataset hosted in a MongoDB
database. This method is more difficult because it requires connection to a MongoDB database. 
However, it is necessary for performing an analysis if the goal is to perform in on all possible
molecule pairs in a large dataset (where the memory requirement would be too large to use the previously
described method). 

First, you must generate the MongoDB database collections. 
```python
from similarities.generate_db import *
d_percent, a_percent, t_percent = 0.10, 0.10, 0.10

# Create molecules MongoDB collection 
all_d = load_mols_db("path_to_dataset.pkl")

# Create molecule pairs MongoDB collection and all possible pair combination IDs. 
add_pairs_db_idx()

# Generate similarity and property difference data for all the molecule pairs IDs. 
create_pairs_db_parallel(verbose=2, sim_min=0.15)
```

Now, you may use this database to perform the KDE analysis on the entire dataset. 
```python
from similarities.kde_analysis import *

all_anal = SimilarityPairsDBAnalysis(anal_percent=0.10, top_percent=0.10, verbose=0)

# OPTIONAL: If you want to improve the computational efficiency of the KDE analysis, generate
# divides for all the similarity measures. This simply identifies the similarity value that divides
# the top_percent similarities from the rest. It can be relatively expensive to perform again and again. 
all_anal.gen_all_divides()

# Perform KDE analysis 
kde_results = all_anal.kde_all()

# Plot results 
ax = all_anal.plot_avg_df(kde_results, return_plot=True, red_labels=True, ratio_name="MongoDB", anal_name=f"all_{all_anal.perc_name}")
```

You may also perform the random sampling analysis with the MongoDB database data. 
```python
from similarities.kde_analysis import *

sim_anal = SimilarityPairsDBAnalysis(anal_percent=0.10, top_percent=0.1, verbose=0)

kde_df = sim_anal.random_sampling(size=100000, num_trials=20, plot=True, return_plot=False, upper_bound=0.172)
```

## Neighborhood Ratios
The neighborhood ratios analysis is derived from [Patterson et. al.](https://doi.org/10.1021/jm960290n). 

This analysis is performed much the same way as the KDE analysis. The only difference is that you must
set the `random_sampling` key word argument `method` to equal `"nhr"`. 
```python
from similarities.kde_analysis import *


sim_anal = SimilarityAnalysisRand(anal_percent=1, top_percent=1, verbose=1)
nhr_df = sim_anal.random_sampling(size=100000, num_trials=10, plot=True, method="nhr", return_plot=False,lower_bound=1.75)

```

Note that boundaries can be calculated the same was described above where `method="nhr"`.
Note also that either the `SimilarityPairsDBAnalysis` or the `SimilarityAnalysisRand` class may be used.


## Ranking Analysis
The ranking analysis measures how many of the top-ranked pairs according to similarity overlap
with the top-ranked pairs according to property difference. 

This analysis is performed much the same way as the KDE analysis. Again, the only difference is that you must
set the `random_sampling` key word argument `method` to equal `"ranking"`. 
```python
from similarities.kde_analysis import *


sim_anal = SimilarityAnalysisRand(anal_percent=1, top_percent=1, verbose=1)
nhr_df = sim_anal.random_sampling(size=100000, num_trials=10, plot=True, method="ranking", return_plot=False,lower_bound=1.75)

```

Note that boundaries can be calculated the same was described above where `method="ranking"`. 
Note also that either the `SimilarityPairsDBAnalysis` or the `SimilarityAnalysisRand` class may be used.

