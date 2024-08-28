import logging
import os
import pickle
import tempfile
import warnings
from argparse import ArgumentParser
from datetime import datetime
from typing import Tuple, TypeAlias

import mlflow
import mlflow.models.signature
import pandas as pd
from dotenv import load_dotenv
from score import evaluate_model
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

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
    grid_search = GridSearchCV(
        model,
        params_grid,
        cv=5,
        scoring=eval_criterion,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def log_runs(
    gridsearch: GridSearchCV,
    model_name: str,
    experiment_id: str,
    log_best_only=False,
    nested_run=False,
    tags={},
):
    """Logging of cross validation results to mlflow tracking server

    Parameters
    ----------
    experiment_name : str
        experiment name
    model_name : str
        Name of the model
    run_index : int
        Index of the run (in Gridsearch)
    conda_env : str
        Dictionary that describes the conda environment (MLFlow Format)
    tags : dict
        Dictionary of extra data and tags (usually features)
    """

    cv_results = gridsearch.cv_results_
    for run_index in range(len(gridsearch.cv_results_["params"])):

        if log_best_only and run_index != gridsearch.best_index_:
            continue

        with mlflow.start_run(
            run_name=model_name + "_" + str(run_index),
            experiment_id=experiment_id,
            nested=nested_run,
        ) as run:

            mlflow.log_param("folds", gridsearch.cv)

            # Logging parameters
            params = list(gridsearch.param_grid.keys())
            for param in params:
                mlflow.log_param(
                    param, cv_results["param_%s" % param][run_index]
                )
            # Logging metrics
            for score_name in [
                score for score in cv_results if "mean_test" in score
            ]:
                mlflow.log_metric(
                    score_name, cv_results[score_name][run_index]
                )
                mlflow.log_metric(
                    score_name.replace("mean", "std"),
                    cv_results[score_name.replace("mean", "std")][run_index],
                )

            # Logging best model
            if run_index == gridsearch.best_index_:
                mlflow.sklearn.log_model(
                    gridsearch.best_estimator_, model_name
                )
                best_run_id = run.info.run_id
                best_run_name = run.info.run_name

                # Logging CV results matrix
                tempdir = tempfile.TemporaryDirectory().name
                os.mkdir(tempdir)
                timestamp = (
                    datetime.now().isoformat().split(".")[0].replace(":", ".")
                )
                filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
                cv_csv = os.path.join(tempdir, filename)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.DataFrame(cv_results).to_csv(cv_csv, index=False)

                mlflow.log_artifact(cv_csv, "cv_results")

                # Logging extra data related to the experiment
                mlflow.set_tags(tags)

    # experiment_id = run.info.experiment_id
    # mlflow.end_run()
    # print(mlflow.get_artifact_uri())
    print("Best model runID:", best_run_id)
    print("Best model runName:", best_run_name)


if __name__ == "__main__":
    DEFAULT_TRAIN_FILE = os.path.join(
        PROJECT_ROOT, "data", "processed", "train.csv"
    )
    DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--train-data-path", type=str, default=DEFAULT_TRAIN_FILE
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
    housing_prepared = pd.read_csv(args.train_data_path)
    housing_labels = housing_prepared["median_house_value"].to_numpy()
    housing_prepared = housing_prepared.drop(
        "median_house_value", axis=1
    ).to_numpy()

    logger.info("Training data loaded from - %s", args.train_data_path)

    # Start mlflow tracking
    mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))

    experiemnt = mlflow.get_experiment_by_name("HPP_model_training_v2")
    if experiemnt:
        experiment_id = experiemnt.experiment_id
    else:
        experiment_id = mlflow.create_experiment(name="HPP_model_training_v2")

    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(
        run_name="Linear_Regression", experiment_id=experiment_id
    ):
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

        mlflow.log_params(params=lin_reg.get_params())
        mlflow.log_metrics({"mae": lin_mae, "rmse": lin_rmse})
        model_sign = mlflow.models.signature.infer_signature(
            model_input=housing_prepared,
            model_output=lin_reg.predict(housing_prepared),
        )
        mlflow.sklearn.log_model(lin_reg, "model", signature=model_sign)

    # train a decision tree model
    with mlflow.start_run(
        run_name="Decision_tree", experiment_id=experiment_id
    ):
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

        mlflow.log_params(params=tree_reg.get_params())
        mlflow.log_metrics({"mae": tree_mae, "rmse": tree_rmse})
        model_sign = mlflow.models.signature.infer_signature(
            model_input=housing_prepared,
            model_output=tree_reg.predict(housing_prepared),
        )
        mlflow.sklearn.log_model(tree_reg, "model", signature=model_sign)

    # Grid search with Random Forest
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = {
        "bootstrap": [True, False],
        "n_estimators": [3, 10],
        "max_features": [2, 6],
    }

    forest_reg_grid = grid_search_tuning(
        forest_reg,
        housing_prepared,
        housing_labels,
        params_grid=param_grid,
        eval_criterion="neg_mean_squared_error",
    )

    log_runs(
        forest_reg_grid,
        "Random_Forest",
        experiment_id=experiment_id,
        log_best_only=True,
    )

    final_model = forest_reg_grid.best_estimator_
    # save the best model
    with open(args.model_path, mode="wb") as f:
        pickle.dump(final_model, f)
    logger.info(
        "Best model, %s, saved to directory: %s",
        final_model.__class__.__name__,
        args.model_path,
    )
