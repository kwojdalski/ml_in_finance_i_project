from kedro.pipeline import Pipeline, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            # node(
            #     func=compare_passenger_capacity_exp,
            #     inputs="preprocessed_shuttles",
            #     outputs="shuttle_passenger_capacity_plot_exp",
            # ),
            # node(
            #     func=compare_passenger_capacity_go,
            #     inputs="preprocessed_shuttles",
            #     outputs="shuttle_passenger_capacity_plot_go",
            # ),
            # node(
            #     func=create_confusion_matrix,
            #     inputs="companies",
            #     outputs="dummy_confusion_matrix",
            # ),
        ]
    )
