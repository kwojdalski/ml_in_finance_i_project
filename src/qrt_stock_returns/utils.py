import logging
from pathlib import Path

import pandas as pd
from kedro.config import MissingConfigException
from kedro.framework.project import find_pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
project_path = Path(__file__).parent.parent.parent

bootstrap_project(project_path)

with KedroSession.create(project_path=project_path) as session:
    context = session.load_context()
    catalog = context.catalog
    pipelines = find_pipelines()

try:
    conf_params = context.config_loader.get("parameters")
except MissingConfigException:
    conf_params = {}

id_cols = conf_params["raw_data"]["id_cols"]
cat_cols = conf_params["raw_data"]["cat_cols"]
kfold = conf_params["model_options"]["kfold"]


def get_node_idx(pipeline, node_name):
    """Get the index of a node in a pipeline by its name.

    Args:
        pipeline: The pipeline to search in
        node_name: Name of the node to find

    Returns:
        int: Index of the node in the pipeline
    """
    return next(i for i, node in enumerate(pipeline.nodes) if node.name == node_name)


def get_node_names(pipeline):
    """Get list of all node names in a pipeline.

    Args:
        pipeline: The pipeline to get node names from

    Returns:
        list: List of node names in the pipeline
    """
    return [node.name for node in pipeline.nodes]


def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict):
    """Run a specific node from a pipeline.

    Args:
        pipeline_name: Name of the pipeline
        node_name: Name of the node to run
        inputs: Dictionary of input parameters for the node

    Returns:
        Output from running the node
    """
    node_idx = get_node_idx(pipelines[pipeline_name], node_name)
    return pipelines[pipeline_name].nodes[node_idx].run(inputs)


def _handle_dataframe_io(filepath: str, df=None, mode="read"):
    """Handle reading/writing dataframes in different formats.

    Args:
        filepath: Path to the data file
        df: DataFrame to save (only needed for write mode)
        mode: Either 'read' or 'write'

    Returns:
        DataFrame if reading, None if writing
    """
    if filepath.endswith(".csv"):
        print(f"{'Loading' if mode=='read' else 'Saving'} CSV file...")
        return (
            pd.read_csv(filepath)
            if mode == "read"
            else df.to_csv(filepath, index=False)
        )
    elif filepath.endswith(".parquet"):
        print(f"{'Loading' if mode=='read' else 'Saving'} Parquet file...")
        return (
            pd.read_parquet(filepath)
            if mode == "read"
            else df.to_parquet(filepath, index=False)
        )
    elif filepath.endswith(".pickle") or filepath.endswith(".pkl"):
        print(f"{'Loading' if mode=='read' else 'Saving'} Pickle file...")
        return pd.read_pickle(filepath) if mode == "read" else df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file extension for {filepath}")


def get_node_inputs(node, catalog):
    """Get input paths for a pipeline node and load the data.

    Args:
        node: Pipeline node to get inputs for
        catalog: Data catalog containing dataset info

    Returns:
        dict: Dictionary mapping input names to their loaded dataframes
    """
    print(f"Getting inputs for node: {node.name}")
    print(f"Node inputs: {node.inputs}")

    inputs = {}
    for input_name in node.inputs:
        if isinstance(input_name, str):
            print(f"\nProcessing input: {input_name}")
            if not input_name.startswith("params:"):
                try:
                    filepath = str(catalog.datasets[input_name]._filepath)
                    print(f"Loading from filepath: {filepath}")

                    df = _handle_dataframe_io(filepath, mode="read")
                    print(f"Successfully loaded dataframe with shape: {df.shape}")
                    inputs[input_name] = df
                except Exception as e:
                    print(f"Error loading data for {input_name}: {str(e)}")
                    inputs[input_name] = None
            else:
                print(f"Skipping parameter input: {input_name}")

    print(f"\nFinished loading {len(inputs)} inputs")
    return inputs


def get_node_outputs(node, catalog):
    """Get output data from a pipeline node.

    Args:
        node: Pipeline node to get outputs for
        catalog: Data catalog containing dataset info
        outputs: Dictionary of output data to save

    Returns:
        dict: Dictionary mapping output names to their loaded dataframes
    """
    print(f"Getting outputs for node: {node.name}")
    print(f"Node outputs: {node.outputs}")

    output_dict = {}
    for output_name in node.outputs:
        if isinstance(output_name, str):
            print(f"\nProcessing output: {output_name}")
            try:
                filepath = str(catalog.datasets[output_name]._filepath)
                print(f"Loading from filepath: {filepath}")

                df = _handle_dataframe_io(filepath, mode="read")
                print(f"Successfully loaded dataframe with shape: {df.shape}")
                output_dict[output_name] = df
            except Exception as e:
                print(f"Error loading data for {output_name}: {str(e)}")
                output_dict[output_name] = None

    print(f"\nFinished loading {len(node.outputs)} outputs")
    return output_dict


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
