from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_statistical_features,
    calculate_technical_indicators,
    drop_id_cols,
    drop_missing_returns,
    drop_obsolete_technical_indicators,
    filter_infinity_values,
    load_data,
    preprocess_data,
    remove_duplicated_columns,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs=["x_train_raw", "y_train_raw", "x_test_raw"],
                outputs=["train_df", "test_df"],
                name="load_data_node",
            ),
            node(
                func=drop_missing_returns,
                inputs=["train_df"],
                outputs="train_df_dropped",
                name="drop_missing_returns_node",
            ),
            node(
                func=preprocess_data,
                inputs=[
                    "train_df_dropped",
                    "test_df",
                    "params:remove_id_cols",
                ],
                outputs=["train_df_preprocessed", "test_df_preprocessed"],
                name="preprocess_data_node",
            ),
            node(
                func=calculate_statistical_features,
                inputs=["train_df_preprocessed", "test_df_preprocessed"],
                outputs=[
                    "train_df_statistical_features",
                    "test_df_statistical_features",
                    "statistical_features",
                ],
                name="calculate_statistical_features_node",
            ),
            node(
                func=calculate_technical_indicators,
                inputs=[
                    "train_df_statistical_features",
                    "test_df_statistical_features",
                ],
                outputs=[
                    "train_ta_indicators",
                    "test_ta_indicators",
                ],
                name="calculate_technical_indicators_node",
            ),
            node(
                func=drop_id_cols,
                inputs=["train_ta_indicators", "test_ta_indicators"],
                outputs=["train_ta_indicators_dropped", "test_ta_indicators_dropped"],
                name="drop_id_cols_node",
            ),
            node(
                func=drop_obsolete_technical_indicators,
                inputs=[
                    "train_ta_indicators_dropped",
                    "params:target",
                ],
                outputs=[
                    "train_df_technical_indicators",
                    "test_df_technical_indicators",
                ],
                name="drop_obsolete_technical_indicators_node",
            ),
            node(
                func=filter_infinity_values,
                inputs=[
                    "train_df_technical_indicators",
                    "test_df_technical_indicators",
                    "params:features",
                    "params:target",
                ],
                outputs=["train_df_filtered", "test_df_filtered"],
                name="filter_infinity_values_node",
            ),
            node(
                func=remove_duplicated_columns,
                inputs=["train_df_filtered", "test_df_filtered", "params:features"],
                outputs=["train_df_rm_duplicates", "test_df_rm_duplicates"],
                name="remove_duplicated_columns_node",
            ),
            node(
                func=split_data,
                inputs=["train_df_rm_duplicates", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
        ]
    )
