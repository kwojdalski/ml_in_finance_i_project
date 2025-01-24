# %%
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.debug("Starting XGBoost")
# %%
try:
    # If we're in VSCode, find the src path this way
    path = Path(__file__).parent.parent
    path = path / "src"
except NameError:
    # If we're in Jupyter, find the path this way instead
    path = Path().absolute().parent

sys.path.append(str(path))

import kedro.ipython
from kedro.ipython import get_ipython

kedro.ipython.load_ipython_extension(get_ipython())

# %%
import sys

import kedro.ipython

from src.qrt_stock_returns.utils import get_node_idx, get_node_outputs


# %% [markdown]
# #### Helper Function
# - Run pipeline nodes right here in the notebook
# %%
def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict):
    """
    Executes a specific node within the data processing pipeline.

    Parameters:
        pipeline_name (str): Target pipeline identifier
        node_name (str): Specific node to execute
        inputs (dict): Node input parameters

    Returns:
        Output from node execution
    """
    node_idx = get_node_idx(pipelines[pipeline_name], node_name)
    return pipelines[pipeline_name].nodes[node_idx].run(inputs)


# %% [markdown]
# ### Loading Up Kedro Config
# Grab all our parameters from the config file
# * This has elements like our target variable and k-fold settings
# %%
conf_params = context.config_loader.get("parameters")
target = conf_params["model_options"]["target"]
kfold = conf_params["model_options"]["kfold"]

# %%

log.debug("Loading up Kedro Config")
# %% [markdown]
# ### Getting Data from the Data Processing Pipeline
# %%
out9 = get_node_outputs(
    pipelines["data_processing"].nodes[
        get_node_idx(pipelines["data_processing"], "remove_duplicates_and_nans_node")
    ],
    catalog,
)

# %% [markdown]
# ## Split Data into Training and Test Sets
# %%
out10 = run_pipeline_node(
    "data_science",
    "split_data_node",
    {
        "train_df_clean": out9["train_df_clean"],
        "params:model_options": conf_params["model_options"],
    },
)
# %%
X_train = out10["X_train"]
X_test = out10["X_test"]
y_train = out10["y_train"]
y_test = out10["y_test"]


# %%
