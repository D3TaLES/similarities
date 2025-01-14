<p align="center">
  <img width="379" alt="image" src="https://github.com/user-attachments/assets/8039650e-26e4-4009-9e46-7c9a49bfdf8a">
</p>


KDE analysis method for evaluating molecular fingerprinting methods and similarity functions with electronic structure data. 
**Modules**: 
* `kde_analysis`: 
* `neighborhood_ratios`: 
* `generate_db`: 

# Installation 


## Settings

# Quickstart
First, install this git repository and configure your `settings.py` file. 
Then you may begin. Start by loading your data as a pandas DataFrame before creating 
an analysis object and performing the analysis. Note that the DataFrame should contain
columns with the name of each electronic property defined in the `settings.py` file as well as a `smiles` column. 

This analysis will randomly pull `num_trials` samples of size `size` from your data and perform the KDE analysis. 
```python
import pandas as pd 
from similarities.analysis import SimilarityAnalysisRand

# Load data
all_d = pd.read_pickle("file_path")

# Create analysis object
sim_anal = SimilarityAnalysisRand(anal_percent=0.20, top_percent=0.1, orig_df=all_d, anal_name="TEST")

# Perform analysis 
kde_df = sim_anal.random_sampling(size=100000, num_trials=10, plot=True, return_plot=False)
```

# More Detailed Docs
## KDE Analysis 
### Using a local data file and performing random sampling
**When to Use**: This method is used to perform the analysis on random samples from a dataset.  
This will be the best option for most use cases. 

First, you must load your dataset as a pandas DataFrame. Note that the DataFrame should contain
columns with the name of each electronic property defined in the `settings.py` file as well as a `smiles` column.
```python
import pandas as pd 

all_d = pd.read_pickle("file_path")
```

Next, create an analysis object and perform the analysis. 
```python
from similarities.analysis import SimilarityAnalysisRand

# Create analysis object
sim_anal = SimilarityAnalysisRand(anal_percent=0.20, top_percent=0.1, verbose=0, orig_df=all_d, anal_name="RandTEST")

# Perform analysis 
kde_df = sim_anal.random_sampling(size=100000, num_trials=10, plot=True, return_plot=False, upper_bound=0.172)
```

Note that the analysis instructions included the key word argument `upper_bound`. This upper bound can be found as 
described in the next section. 

#### Boundaries 


### Using a MongoDB database
**When to Use**: 

#### Boundaries 

## Neighborhood Ratios
The neighborhood ratios analysis is derived from [Patterson et. al.](). 


#### Boundaries 


