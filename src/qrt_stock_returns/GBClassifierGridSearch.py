import logging as log

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# %% 1) tuning n_estimators
class GBClassifierGridSearch:
    def __init__(self, default_params=None):
        """Initialize grid search for gradient boosting classifiers.

        Args:
            default_params (dict): Default parameters for the classifier
        """
        self.default_params = default_params or {
            "learning_rate": 0.1,
            "max_features": "sqrt",
            "subsample": 0.8,
            "max_depth": 8,
            "random_state": 10,
            "min_samples_split": 500,
            "min_samples_leaf": 50,
            "n_estimators": 100,
        }

        self.classifier_class = GradientBoostingClassifier

    def run(self, x_train, y_train):
        """Run basic model with default parameters"""
        log.info(
            f"Running basic {self.classifier_class.__name__} model with default parameters"
        )
        self.model = self.classifier_class(**self.default_params)
        self.model.fit(x_train, y_train)

    def tune_n_estimators(self, x_train, y_train, best_params=None):
        """Tune n_estimators parameter"""
        log.info("tuning n_estimators")
        params = {"n_estimators": range(30, 81, 10)}

        estimator = self.classifier_class(
            **{**self.default_params, **(best_params or {})}
        )
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=1, verbose=False
        )
        return self._run_grid_search(grid, x_train, y_train)

        # Results from previous run:
        # 0.532319 (0.001552) with: {'n_estimators': 30}
        # 0.534089 (0.001092) with: {'n_estimators': 40}
        # 0.534662 (0.001791) with: {'n_estimators': 50}
        # 0.535451 (0.002343) with: {'n_estimators': 60}
        # 0.535757 (0.002051) with: {'n_estimators': 70}
        # 0.536678 (0.001270) with: {'n_estimators': 80}

        def tune_tree_params(self, x_train, y_train, best_params=None):
            """Tune max_depth and min_samples_split parameters"""

        log.info("tuning max_depth and min_sample_split")
        params = {
            "max_depth": range(5, 16, 2),
            "min_samples_split": range(400, 1001, 200),
        }

        estimator = self.classifier_class(**self.default_params, **best_params)
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
        )
        return self._run_grid_search(grid, x_train, y_train)

    def tune_leaf_params(self, x_train, y_train, best_params):
        """Tune min_samples_leaf parameter"""
        log.info("tuning min_samples_leaf")
        params = {
            "min_samples_leaf": range(40, 70, 10),
            "min_samples_split": range(400, 1001, 200),
        }

        estimator = self.classifier_class(**{**self.default_params, **best_params})
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
        )
        return self._run_grid_search(grid, x_train, y_train)

    def tune_max_features(self, x_train, y_train, best_params):
        """Tune max_features parameter"""
        log.info("tuning max_features")
        params = {"max_features": range(7, 20, 2)}

        estimator = self.classifier_class(**{**self.default_params, **best_params})
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=1, verbose=True
        )
        return self._run_grid_search(grid, x_train, y_train)

    def _run_grid_search(self, grid, x_train, y_train):
        """Helper method to run grid search and log results"""
        grid_result = grid.fit(x_train, y_train)

        log.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params = grid_result.cv_results_["params"]

        for mean, stdev, param in zip(means, stds, params):
            log.info(f"{mean} ({stdev}) with: {param}")

        return grid_result


class HistGBClassifierGridSearch(GBClassifierGridSearch):
    def __init__(self, default_params=None):
        """Initialize grid search for histogram-based gradient boosting classifiers.

        Args:
            default_params (dict): Default parameters for the classifier
        """

        self.default_params = default_params or {
            "learning_rate": 0.1,
            "max_bins": 255,
            "l2_regularization": 0,
            "random_state": 10,
        }
        self.classifier_class = HistGradientBoostingClassifier

    def tune_n_estimators(self, x_train, y_train, best_params=None):
        """Tune max_iter parameter (equivalent to n_estimators)"""
        log.info("tuning max_iter")
        params = {"max_iter": range(30, 81, 10)}

        estimator = self.classifier_class(
            **{**self.default_params, **(best_params or {})}
        )
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=1, verbose=False
        )
        return self._run_grid_search(grid, x_train, y_train)

    def tune_tree_params(self, x_train, y_train, best_params):
        """Tune max_depth and min_samples_leaf parameters"""
        log.info("tuning max_depth and min_samples_leaf")
        params = {
            "max_depth": range(5, 16, 2),
            "min_samples_leaf": range(20, 51, 10),
        }

        estimator = self.classifier_class(**{**self.default_params, **best_params})
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
        )
        return self._run_grid_search(grid, x_train, y_train)

    def tune_leaf_params(self, x_train, y_train, best_params):
        """Tune l2_regularization parameter"""
        log.info("tuning l2_regularization")
        params = {"l2_regularization": [0, 0.1, 0.5, 1.0, 2.0]}

        estimator = self.classifier_class(**{**self.default_params, **best_params})
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
        )
        return self._run_grid_search(grid, x_train, y_train)

    def tune_max_features(self, x_train, y_train, best_params):
        """Tune max_bins parameter (equivalent to max_features)"""
        log.info("tuning max_bins")
        params = {"max_bins": [128, 255, 512]}

        estimator = self.classifier_class(**{**self.default_params, **best_params})
        grid = GridSearchCV(
            estimator, params, cv=5, scoring="accuracy", n_jobs=1, verbose=True
        )
        return self._run_grid_search(grid, x_train, y_train)
