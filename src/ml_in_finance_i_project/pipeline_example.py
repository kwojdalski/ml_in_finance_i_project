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
                ],
                outputs=["train_df_preprocessed", "test_df_preprocessed"],
                name="preprocess_data_node",
            ),
            node(
                func=calculate_technical_indicators,
                inputs=["train_df_preprocessed", "test_df_preprocessed"],
                outputs=[
                    "train_ta_indicators",
                    "test_ta_indicators",
                ],
                name="calculate_technical_indicators_node",
            ),
            node(
                func=drop_obsolete_technical_indicators,
                inputs=[
                    "train_ta_indicators",
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
