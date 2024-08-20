import os
import shutil
import tarfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.house_price_predictor.ingest_data import (
    fetch_housing_data,
    load_housing_data,
    split_data,
)

# Constants for the test
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/\
                datasets/housing/housing.tgz"
HOUSING_PATH = (
    r"artifacts/housing"  # Changed to a directory path, use raw string
)


@pytest.fixture(scope="module")
def housing_data_setup():
    if not os.path.exists(HOUSING_PATH):
        os.makedirs(HOUSING_PATH, exist_ok=True)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")

    # Creating a dummy tar.gz file with a single csv file inside
    with tarfile.open(tgz_path, "w:gz") as tar:
        data = pd.DataFrame(
            {
                "longitude": [-122.23, -122.22, -122.25, -122.24],
                "latitude": [37.88, 37.86, 37.85, 37.87],
                "housing_median_age": [41, 21, 35, 30],
                "total_rooms": [880, 7099, 1500, 2000],
                "total_bedrooms": [129, 1106, 300, 400],
                "population": [322, 2401, 500, 600],
                "households": [126, 1138, 200, 300],
                "median_income": [8.3252, 8.3014, 7.2574, 5.6431],
                "median_house_value": [452600, 358500, 342200, 269700],
            }
        )

        csv_path = os.path.join(HOUSING_PATH, "housing.csv")
        data.to_csv(csv_path, index=False)
        tar.add(csv_path, arcname="housing.csv")
        os.remove(csv_path)

    yield tgz_path

    # Teardown after test
    shutil.rmtree(
        HOUSING_PATH
    )  # Change to rmtree to ensure the entire directory is deleted


@pytest.mark.usefixtures("housing_data_setup")
@patch("urllib.request.urlretrieve")
def test_ingest_data(mock_urlretrieve):
    # Mock the urlretrieve to not perform real HTTP requests
    mock_urlretrieve.return_value = (None, None)

    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)
    train_set, test_set = split_data(housing, test_size=0.2, seed=42)

    assert not housing.empty, "DataFrame should not be empty"
    assert (
        "longitude" in housing.columns
    ), "Longitude should be one of the columns"
    assert len(train_set) > 0, "Training set should not be empty"
    assert len(test_set) > 0, "Test set should not be empty"
    assert len(train_set) > len(
        test_set
    ), "Training set should be larger than test set"
