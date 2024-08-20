import logging
import os
import pickle
from argparse import ArgumentParser
from typing import Tuple, TypeAlias

import numpy as np
import pandas as pd
from scipy.stats import randint
from score import evaluate_model
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = "/mnt/c/Users/satyendra.mishra/Work/mle-training/"

Regressor: TypeAlias = RegressorMixin

logger = logging.getLogger(__name__)


def train_model(
    model: Regressor, X_train: pd.DataFrame, y_train: pd.DataFrame
) -> Regressor:
    """
    trains the model on given data

    Parameters
    ----------
    model : Regressor
        model to train
    X_train : pd.DataFrame
        training data without labels
    y_train : pd.DataFrame
        labels for training data

    Returns
    -------
    model : Regressor
        trained model

    """
    logger.info("Model training initiated - %s", model.__class__.__name__)
    model.fit(X_train, y_train)
    logger.info("Model training finished - %s", model.__class__.__name__)
    return model


def randomized_search_tuning(
    model: Regressor,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params_dist: dict,
    eval_criterion: str = "neg_mean_squared_error",
) -> Tuple[Regressor, dict, float]:
    """
    Performs randomized search hyperparamter tuning of the model

    Paramaters
    ----------
    model : Regressor
        model to tune
    X_train : pd.DataFrame
        training dataset without labels
    y_train : pd.Series
        labels of trining data
    params_dist : dict
        dictionary of hyperparameters
    eval_criterion : str
        evaluations strategy

    Returns
    -------
    A tuple of tuned model, best hyperparameters, best score

    """

    logger.info(
        "Initiating Hyperparameter tuning with Randomized search on %s",
        model.__class__.__name__,
    )
    rnd_search = RandomizedSearchCV(
        model,
        param_distributions=params_dist,
        n_iter=10,
        cv=5,
        scoring=eval_criterion,
        random_state=42,
    )
    rnd_search.fit(X_train, y_train)
    logger.info("Randomized search completed.")
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    logger.info(
        "Best params = %s, Best score = %f",
        rnd_search.best_params_,
        rnd_search.best_score_,
    )
    return (
        rnd_search.best_estimator_,
        rnd_search.best_params_,
        rnd_search.best_score_,
    )


def grid_search_tuning(
    model: Regressor,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params_grid: dict,
    eval_criterion: str = "neg_mean_squared_error",
) -> Tuple[Regressor, dict, float]:
    """
    Performs Grid search hyperparamter tuning of the model

    Paramaters
    ----------
    model : Regressor
        model to tune
    X_train : pd.DataFrame
        training dataset without labels
    y_train : pd.Series
        labels of trining data
    params_dist : dict
        dictionary of hyperparameters
    eval_criterion : str
        evaluations strategy

    Returns
    -------
    A tuple of tuned model, best hyperparameters, best score

    """
    logger.info(
        "Initiating Hyperparameter tuning with Grid search on %s",
        model.__class__.__name__,
    )
    grid_search = GridSearchCV(
        model,
        params_grid,
        cv=5,
        scoring=eval_criterion,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    logger.info("Grid search completed.")
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    logger.info(
        "Best params = %s, Best score = %f",
        grid_search.best_params_,
        grid_search.best_score_,
    )
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


if __name__ == "__main__":
    DEFAULT_TRAIN_PATH = os.path.join(
        PROJECT_ROOT, "data", "processed", "train.csv"
    )
    DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--train-data-path", type=str, default=DEFAULT_TRAIN_PATH
    )
    parser.add_argument(
        "-m", "--model-path", type=str, default=DEFAULT_MODEL_PATH
    )

    args = parser.parse_args()

    assert os.path.exists(
        args.train_data_path
    ), "Specified path for training data not found"
    assert os.path.exists(
        args.model_path
    ), "Specified path for model not found"

    # load the training data
    housing_df = pd.read_csv(args.train_data_path)
    logger.info("Training data loaded from - %s", args.train_data_path)
    # separate the features and labels
    housing_prepared = housing_df.drop("median_house_value", axis=1).to_numpy()
    housing_labels = housing_df["median_house_value"].to_numpy()

    # train a linear regression model
    lin_reg = LinearRegression()
    lin_reg = train_model(lin_reg, housing_prepared, housing_labels)
    lin_mae, lin_rmse = evaluate_model(
        lin_reg, housing_prepared, housing_labels
    )
    logger.info(
        "Evaluation on training data: mae = %.2f, mse = %.2f",
        lin_mae,
        lin_rmse,
    )
    # train a decision tree model
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg = train_model(tree_reg, housing_prepared, housing_labels)
    tree_mae, tree_rmse = evaluate_model(
        tree_reg, housing_prepared, housing_labels
    )
    logger.info(
        "Evaluation on training data: mae = %.2f, mse = %.2f",
        tree_mae,
        tree_rmse,
    )
    # randomized search with Random forest
    forest_reg = RandomForestRegressor(random_state=42)
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    rnd_best_model, rnd_best_params, rnd_best_score = randomized_search_tuning(
        forest_reg,
        housing_prepared,
        housing_labels,
        params_dist=param_distribs,
        eval_criterion="mean_squared_error",
    )

    # Grid search with Random Forest
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    # train across 5 folds, that's a total of
    # (12+6)*5=90 rounds of training
    grid_best_model, grid_best_params, grid_best_score = grid_search_tuning(
        forest_reg,
        housing_prepared,
        housing_labels,
        params_grid=param_grid,
        eval_criterion="mean_squard_error",
    )

    if grid_best_score > rnd_best_score:
        feature_importances = grid_best_model.feature_importances_
        sorted(
            zip(feature_importances, housing_prepared.columns), reverse=True
        )
        final_model = grid_best_model
    else:
        feature_importances = rnd_best_model.feature_importances_
        sorted(
            zip(feature_importances, housing_prepared.columns), reverse=True
        )
        final_model = rnd_best_model

    # save the best model
    with open(args.model_path, mode="wb") as f:
        pickle.dump(final_model, f)
    logger.info(
        "Best model, %s, saved to directory: %s",
        final_model.__class__.__name__,
        args.model_path,
    )
