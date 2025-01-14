from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    feature_importance,
    plot_correlation_matrix,
    plot_model_accuracy,
    plot_na,
    plot_nan_percentages,
    plot_ret_and_vol,
    simulate_strategy,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline that generates visualization plots for reporting"""
    return pipeline(
        [
            node(
                func=plot_correlation_matrix,
                inputs="train_df",
                outputs="correlation_matrix_plot",
                name="plot_correlation_matrix_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=feature_importance,
                inputs=["model", "features", "params:feature_importance_threshold"],
                outputs="feature_importance_plot",
                name="plot_feature_importance_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=plot_nan_percentages,
                inputs="cleaned_train",
                outputs="nan_percentages_plot",
                name="plot_nan_percentages_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=plot_na,
                inputs="train_df",
                outputs="na_distribution_plot",
                name="plot_na_distribution_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=plot_model_accuracy,
                inputs="model_results",
                outputs="model_accuracy_plot",
                name="plot_model_accuracy_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=plot_ret_and_vol,
                inputs=["train_df", "params:example_row_id"],
                outputs="returns_volume_plot",
                name="plot_returns_volume_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=simulate_strategy,
                inputs=["y_test", "y_predict", "params:n_simulations", "params:n_days"],
                outputs="strategy_simulation_plot",
                name="simulate_strategy_node",
                tags=["reporting", "visualization"],
            ),
        ]
    )
