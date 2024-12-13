from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_models,
    train_decision_tree,
    train_gradient_boosting,
    train_neural_network,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_decision_tree,
                inputs=["X_train", "parameters"],
                outputs="dt_model",
                name="train_decision_tree_node",
            ),
            node(
                func=train_gradient_boosting,
                inputs=["X_train", "parameters"],
                outputs="gb_model",
                name="train_gradient_boosting_node",
            ),
            node(
                func=train_neural_network,
                inputs=["X_train", "parameters"],
                outputs="nn_model",
                name="train_neural_network_node",
            ),
            node(
                func=evaluate_models,
                inputs=["dt_model", "gb_model", "nn_model", "processed_test_df"],
                outputs="model_evaluation",
                name="evaluate_models_node",
            ),
        ]
    )
