import os
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from src.house_price_predictor.train import train_and_save_models

warnings.filterwarnings(
    "ignore", message=".*utcfromtimestamp.*", category=DeprecationWarning
)


# Constants for the test
DATA_PATH = "test_data/train.csv"
MODEL_OUTPUT_PATH = "test_models"


def setup_module(module):
    """Setup for the entire module"""
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    # Create a dummy dataset
    data = pd.DataFrame(
        {
            "longitude": np.random.uniform(-124, -112, 10),
            "latitude": np.random.uniform(32, 42, 10),
            "housing_median_age": np.random.randint(1, 50, 10),
            "total_rooms": np.random.randint(100, 1000, 10),
            "total_bedrooms": np.random.randint(20, 500, 10),
            "population": np.random.randint(100, 10000, 10),
            "households": np.random.randint(50, 500, 10),
            "median_income": np.random.uniform(1.5, 10.0, 10),
            "median_house_value": np.random.randint(100000, 500000, 10),
            "ocean_proximity": [
                "NEAR BAY",
                "INLAND",
                "NEAR OCEAN",
                "INLAND",
                "NEAR BAY",
                "INLAND",
                "NEAR BAY",
                "INLAND",
                "NEAR BAY",
                "INLAND",
            ],
        }
    )

    data.to_csv(DATA_PATH, index=False)


def teardown_module(module):
    """Teardown for the entire module"""
    shutil.rmtree(MODEL_OUTPUT_PATH)
    os.remove(DATA_PATH)


def test_train_and_save_models():
    """Test the entire training and saving pipeline"""
    train_and_save_models(DATA_PATH, MODEL_OUTPUT_PATH)

    # Check if the models were saved correctly
    expected_models = [
        "linear_regression.pkl",
        "decision_tree.pkl",
        "random_forest.pkl",
        "random_forest_tuned.pkl",
    ]
    saved_models = os.listdir(MODEL_OUTPUT_PATH)
    assert set(expected_models) == set(
        saved_models
    ), "Not all models were saved correctly"

    # Optionally, you can load the models to assert their type or properties
    for model_name in expected_models:
        model_path = os.path.join(MODEL_OUTPUT_PATH, model_name)
        model = joblib.load(model_path)
        assert model is not None, f"Model {model_name} failed to load"


if __name__ == "__main__":
    pytest.main()
