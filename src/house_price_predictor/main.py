import os
import tempfile
from datetime import datetime

import mlflow
import pandas as pd
from dotenv import load_dotenv
from ingest_data import (
    get_preprocessing_pipeline,
    load_housing_data,
    split_data,
)
from score import evaluate_model
from sklearn.ensemble import RandomForestRegressor
from train import grid_search_tuning

load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


if __name__ == "__main__":

    mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))

    experiment_name = "Full HPP pipeline"

    experiement = mlflow.get_experiment_by_name(experiment_name)
    if experiement:
        experiement_id = experiement.experiment_id
    else:
        experiement_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(
        run_name="Parent run", experiment_id=experiement_id
    ) as p0_run:

        # child run for data ingestion
        with mlflow.start_run(
            run_name="data_ingestion_and_preprocessing",
            experiment_id=experiement_id,
            nested=True,
        ):
            housing_data = load_housing_data()
            train, test = split_data(housing_data, test_size=0.25)

            mlflow.log_artifact(
                "data/processed/train.csv", artifact_path="data"
            )
            mlflow.log_artifact(
                "data/processed/test.csv", artifact_path="data"
            )

            X_train = train.drop(
                "median_house_value", axis=1
            )  # drop labels for training set

            y_train = train["median_house_value"].copy()

            X_test = test.drop(
                "median_house_value", axis=1
            )  # drop labels for training set

            y_test = test["median_house_value"].copy()

            # actegorical and numerical features
            cat_attribs = ["ocean_proximity"]
            num_attribs = X_train.drop(cat_attribs, axis=1).columns.tolist()

            # lof the feature columns
            mlflow.log_param("Feature columns", X_train.columns.tolist())
            mlflow.log_param("Label column", y_train.name)
            mlflow.log_param("Numerical Features", num_attribs)
            mlflow.log_param("Categorical Features", cat_attribs)

            # preprocessing pipeline
            preprocessing_pipeline = get_preprocessing_pipeline(
                X_train, num_attribs, cat_attribs
            )
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)

        # Child run for Model training: Grid search with Random Forest
        forest_reg = RandomForestRegressor(random_state=42)
        param_grid = {
            "bootstrap": [True, False],
            "n_estimators": [3, 10],
            "max_features": [2, 6],
        }

        forest_reg_grid = grid_search_tuning(
            forest_reg,
            X_train,
            y_train,
            params_grid=param_grid,
            eval_criterion="neg_mean_squared_error",
        )

        cv_results = forest_reg_grid.cv_results_

        with mlflow.start_run(
            run_name="Random_Forest_best_model",
            experiment_id=experiement_id,
            nested=True,
        ):
            mlflow.log_param("CV Folds", forest_reg_grid.cv)
            mlflow.log_params(forest_reg_grid.best_estimator_.get_params())
            mlflow.log_metric(
                "neg_mean_squared_error", forest_reg_grid.best_score_
            )

            mlflow.sklearn.log_model(
                forest_reg_grid.best_estimator_,
                "RandomForestModel",
                input_example=X_train.iloc[[0, 50, 100]],
            )
            # Logging CV results matrix
            tempdir = tempfile.TemporaryDirectory().name
            if not os.path.exists(tempdir):
                os.mkdir(tempdir)
            timestamp = (
                datetime.now().isoformat().split(".")[0].replace(":", ".")
            )
            filename = "%s-%s-cv_results.csv" % (
                "RandomForestModel",
                timestamp,
            )
            cv_csv = os.path.join(tempdir, filename)
            pd.DataFrame(cv_results).to_csv(cv_csv, index=False)

            mlflow.log_artifact(cv_csv, "cv_results")

            train_csv_path = train.to_csv(
                os.path.join(tempdir, "train.csv"), index=False
            )
            train_dataset = mlflow.data.from_pandas(
                train, source=train_csv_path, targets="median_house_value"
            )
            mlflow.log_input(train_dataset, context="training")

        # Child run for Model scoring
        with mlflow.start_run(
            run_name="evaluation", experiment_id=experiement_id, nested=True
        ):
            mae, rmse, r2 = evaluate_model(
                forest_reg_grid.best_estimator_, X_test, y_test
            )
            mlflow.log_metrics(
                {
                    "mean_absolute_error": mae,
                    "root_mean_squared_error": rmse,
                    "r_squared": r2,
                }
            )

            test_csv_path = test.to_csv(
                os.path.join(tempdir, "test.csv"), index=False
            )
            test_dataset = mlflow.data.from_pandas(
                test, source=test_csv_path, targets="median_house_value"
            )
            mlflow.log_input(test_dataset, context="testing")
