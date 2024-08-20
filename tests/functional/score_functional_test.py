import os
import warnings
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Suppress specific DeprecationWarnings about utcfromtimestamp from dateutil.tz
warnings.filterwarnings(
    "ignore",
    message=".*utcfromtimestamp.*",
    category=DeprecationWarning,
    module="dateutil.tz",
)


def create_test_data(temp_dir):
    # Define a sample DataFrame with a 'median_house_value' column
    data = pd.DataFrame(
        {
            "longitude": [-122.23, -122.22],
            "latitude": [37.88, 37.86],
            "housing_median_age": [41, 21],
            "total_rooms": [880, 7099],
            "total_bedrooms": [129, 1106],
            "population": [322, 2401],
            "households": [126, 1138],
            "median_income": [8.3252, 8.3014],
            "ocean_proximity": ["NEAR BAY", "INLAND"],
            "median_house_value": [452600, 358500],
        }
    )

    num_attribs = [
        col
        for col in data.columns
        if data[col].dtype in ["int64", "float64"]
        and (col != "median_house_value")
    ]
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    """Ensure only to drop 'median_house_value' when passing
    data to the transformer"""
    features = full_pipeline.fit_transform(
        data.drop("median_house_value", axis=1)
    )
    labels = data["median_house_value"]

    # Train a RandomForest model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(features, labels)

    model_path = os.path.join(temp_dir, "model.pkl")
    joblib.dump(model, model_path)
    data_path = os.path.join(temp_dir, "data.csv")
    data.to_csv(data_path, index=False)

    return model_path, data_path


@pytest.fixture
def setup_environment():
    with TemporaryDirectory() as temp_dir:
        model_path, data_path = create_test_data(temp_dir)
        output_path = os.path.join(temp_dir, "output")
        os.makedirs(output_path)
        yield model_path, data_path, output_path


def test_score_models(setup_environment):
    model_path, data_path, output_path = setup_environment
    from src.components.score import score_models

    score_models(model_path, data_path, output_path)

    # Validate the creation of prediction files
    predictions_files = [
        f for f in os.listdir(output_path) if f.endswith("_predictions.csv")
    ]
    assert len(predictions_files) > 0, "No prediction files were created"
    for prediction_file in predictions_files:
        full_path = os.path.join(output_path, prediction_file)
        predictions = pd.read_csv(full_path)
        assert not predictions.empty, "Prediction file is empty"
        assert "Predictions" in (
            predictions.columns,
            "Predictions column is missing",
        )
