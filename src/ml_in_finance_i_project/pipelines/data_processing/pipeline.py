from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_statistical_features,
    calculate_technical_indicators,
    drop_id_cols,
    drop_obsolete_technical_indicators,
    filter_infinity_values,
    load_and_preprocess_data,
    remove_duplicated_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_preprocess_data,
                inputs=[
                    "x_train_raw",
                    "y_train_raw",
                    "x_test_raw",
                    "params:remove_id_cols",
                ],
                outputs=["train_df_preprocessed", "test_df_preprocessed"],
                name="load_and_preprocess_data_node",
                tags=["data_loading", "data_cleaning", "data_preprocessing"],
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
                tags=["feature_engineering"],
            ),
            node(
                func=calculate_technical_indicators,
                inputs=[
                    "train_df_statistical_features",
                    "test_df_statistical_features",
                    "params:features_ret_vol",
                ],
                outputs=[
                    "train_ta_indicators",
                    "test_ta_indicators",
                ],
                name="calculate_technical_indicators_node",
                tags=["feature_engineering"],
            ),
            node(
                func=drop_id_cols,
                inputs=["train_ta_indicators", "test_ta_indicators"],
                outputs=["train_ta_indicators_dropped", "test_ta_indicators_dropped"],
                name="drop_id_cols_node",
                tags=["data_cleaning"],
            ),
            node(
                func=drop_obsolete_technical_indicators,
                inputs=[
                    "train_ta_indicators_dropped",
                    "test_ta_indicators_dropped",
                    "params:target",
                ],
                outputs=[
                    "train_df_technical_indicators",
                    "test_df_technical_indicators",
                    "features",
                ],
                name="drop_obsolete_technical_indicators_node",
                tags=["data_cleaning"],
            ),
            node(
                func=filter_infinity_values,
                inputs=[
                    "train_df_technical_indicators",
                    "test_df_technical_indicators",
                    "params:target",
                ],
                outputs=[
                    "train_df_filtered",
                    "test_df_filtered",
                    "features_filtered",
                ],
                name="filter_infinity_values_node",
                tags=["data_cleaning"],
            ),
            node(
                func=remove_duplicated_columns,
                inputs=[
                    "train_df_filtered",
                    "test_df_filtered",
                ],
                outputs=[
                    "train_df_rm_duplicates",
                    "test_df_rm_duplicates",
                    "features_rm_duplicates",
                ],
                name="remove_duplicated_columns_node",
                tags=["data_cleaning"],
            ),
        ]
    )
