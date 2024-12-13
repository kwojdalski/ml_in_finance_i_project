from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_data,
    train_decision_tree,
    train_gradient_boosting,
    train_neural_network,
    tune_decision_tree,
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
                func=train_decision_tree,
                inputs=["X_train", "y_train"],
                outputs="dt_model",
                name="train_decision_tree_node",
                tags=["model_training", "decision_tree"],
            ),
            node(
                func=tune_decision_tree,
                inputs=["X_train", "y_train", "parameters"],
                outputs="tuned_dt_model",
                name="tune_decision_tree_node",
                tags=["model_tuning", "decision_tree"],
            ),
            node(
                func=train_gradient_boosting,
                inputs=["X_train", "y_train", "parameters"],
                outputs="gb_model",
                name="train_gradient_boosting_node",
                tags=["model_training", "gradient_boosting"],
            ),
            node(
                func=train_neural_network,
                inputs=["X_train", "y_train", "parameters"],
                outputs="nn_model",
                name="train_neural_network_node",
                tags=["model_training", "neural_network"],
            ),
        ]
    )
