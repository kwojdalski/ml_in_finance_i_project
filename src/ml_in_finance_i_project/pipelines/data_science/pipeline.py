from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    remove_nan_rows,
    split_data,
    train_decision_tree,
    train_gradient_boosting,
    train_neural_network,
    tune_decision_tree,
    tune_gradient_boosting,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["train_df_rm_duplicates", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["data_splitting"],
            ),
            node(
                func=remove_nan_rows,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs=[
                    "X_train_clean",
                    "X_test_clean",
                    "y_train_clean",
                    "y_test_clean",
                ],
                name="remove_nan_rows_node",
                tags=["data_cleaning"],
            ),
            node(
                func=train_decision_tree,
                inputs=["X_train_clean", "y_train_clean"],
                outputs="base_dt",
                name="train_decision_tree_node",
                tags=["model_training", "decision_tree"],
            ),
            node(
                func=tune_decision_tree,
                inputs=["X_train_clean", "y_train_clean", "parameters"],
                outputs="grid_dt",
                name="tune_decision_tree_node",
                tags=["model_tuning", "decision_tree"],
            ),
            node(
                func=train_gradient_boosting,
                inputs=["X_train_clean", "y_train_clean", "parameters"],
                outputs="base_gb",
                name="train_gradient_boosting_node",
                tags=["model_training", "gradient_boosting"],
            ),
            node(
                func=tune_gradient_boosting,
                inputs=["base_gb", "X_train_clean", "y_train_clean"],
                outputs="tuned_gb",
                name="tune_gradient_boosting_node",
                tags=["model_tuning", "gradient_boosting"],
            ),
            node(
                func=train_neural_network,
                inputs=["X_train_clean", "y_train_clean", "parameters"],
                outputs="nn_model",
                name="train_neural_network_node",
                tags=["model_training", "neural_network"],
            ),
        ]
    )
