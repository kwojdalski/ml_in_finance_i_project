from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_model_results,
    evaluate_xgboost,
    generate_predictions,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_model_accuracy,
    plot_nan_percentages,
    plot_ret_and_vol,
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
                func=plot_feature_importance,
                inputs=[
                    "grid_dt",
                    "X_train",
                    "params:feature_importance_threshold",
                ],
                outputs="feature_importance_plot",
                name="plot_feature_importance_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=plot_nan_percentages,
                inputs="train_df",
                outputs="nan_percentages_plot",
                name="plot_nan_percentages_node",
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
                func=aggregate_model_results,
                inputs=[
                    "base_dt",
                    "grid_dt",
                    "tuned_gb",
                    "nn_model",
                    "X_test",
                    "y_test",
                ],
                outputs=["model_results", "all_metrics"],
                name="aggregate_model_results_node",
                tags=["reporting", "metrics"],
            ),
            node(
                func=evaluate_xgboost,
                inputs=["xgboost_model", "X_test", "y_test", "parameters"],
                outputs="xgboost_metrics",
                name="evaluate_xgboost_node",
                tags=["reporting", "metrics"],
            ),
            node(
                func=generate_predictions,
                inputs=["xgboost_model", "test_df_winsorized"],
                outputs="predictions",
                name="generate_predictions_node",
                tags=["reporting", "predictions"],
            ),
        ]
    )
