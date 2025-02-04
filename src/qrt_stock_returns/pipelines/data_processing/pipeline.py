from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_statistical_features,
    calculate_technical_indicators,
    drop_id_cols,
    drop_obsolete_technical_indicators,
    filter_infinity_values,
    handle_outliers,
    load_data,
    merge_with_features,
    preprocess_data,
    remove_duplicates_and_nans,
    retrieve_id_cols,
    transform_volret_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs=[
                    "x_train_raw",
                    "y_train_raw",
                    "x_test_raw",
                    "params:sample_n",
                ],
                outputs=["train_df", "test_df"],
                name="load_data_node",
                tags=["data_loading", "data_cleaning", "data_preprocessing"],
            ),
            node(
                func=preprocess_data,
                inputs=["train_df", "test_df", "params:drop_na"],
                outputs=["train_df_preprocessed", "test_df_preprocessed"],
                name="preprocess_data_node",
                tags=["data_cleaning", "data_preprocessing"],
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
                func=merge_with_features,
                inputs=[
                    "train_df_preprocessed",
                    "test_df_preprocessed",
                    "train_ta_indicators",
                    "test_ta_indicators",
                    "train_df_statistical_features",
                    "test_df_statistical_features",
                ],
                outputs=["train_df_merged", "test_df_merged"],
                name="merge_with_features_node",
                tags=["feature_engineering"],
            ),
            node(
                func=retrieve_id_cols,
                inputs=["train_df", "test_df"],
                outputs=["train_id_cols", "test_id_cols"],
                name="retrieve_id_cols_node",
                tags=["data_cleaning"],
            ),
            node(
                func=drop_id_cols,
                inputs=["train_df_merged", "test_df_merged"],
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
                func=remove_duplicates_and_nans,
                inputs=[
                    "train_df_filtered",
                    "test_df_filtered",
                    "params:imputing_method",
                ],
                outputs=[
                    "train_df_clean",
                    "test_df_clean",
                ],
                name="remove_duplicates_and_nans_node",
                tags=["data_cleaning"],
            ),
            node(
                func=transform_volret_features,
                inputs=["train_df_clean", "test_df_clean"],
                outputs=["train_df_transformed", "test_df_transformed"],
                name="transform_volret_features_node",
                tags=["feature_engineering"],
            ),
            node(
                func=handle_outliers,
                inputs=[
                    "train_df_transformed",
                    "test_df_transformed",
                    "params:outlier_threshold",
                    "params:outlier_method",
                ],
                outputs=["train_df_winsorized", "test_df_winsorized"],
                name="handle_outliers_node",
                tags=["data_cleaning"],
            ),
        ]
    )
