import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.components.train import (
    load_data,
    preprocess_features,
    randomized_search_cv,
    train_and_save_models,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
)


class TestTrainModule(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "longitude": [-122.23, -122.22],
                "latitude": [37.88, 37.86],
                "housing_median_age": [41, 21],
                "total_rooms": [880, 7099],
                "total_bedrooms": [129, 1106],
                "population": [322, 2401],
                "households": [126, 1138],
                "median_income": [8.3252, 8.3014],
                "median_house_value": [452600, 358500],
                "ocean_proximity": ["NEAR BAY", "NEAR BAY"],
            }
        )

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.sample_data
        data = load_data("dummy_path")
        pd.testing.assert_frame_equal(data, self.sample_data)

    @patch("pandas.read_csv")
    def test_preprocess_features(self, mock_read_csv):
        mock_read_csv.return_value = self.sample_data
        data = load_data("dummy_path")
        prepared, labels = preprocess_features(data)
        self.assertEqual(prepared.shape[0], data.shape[0])
        self.assertIsNotNone(labels)

    @patch("pandas.read_csv")
    @patch("joblib.dump")
    def test_train_and_save_models(self, mock_joblib_dump, mock_read_csv):
        # Ensure the mock for read_csv is set up to return the sample data
        mock_read_csv.return_value = self.sample_data

        # Mock model training functions
        with patch(
            "src.components.train.train_linear_regression",
            return_value="linear_model"
        ), patch(
            "src.components.train.train_decision_tree",
            return_value="tree_model"
        ), patch(
            "src.components.train.train_random_forest",
            return_value="forest_model"
        ), patch(
            "src.components.train.randomized_search_cv",
            return_value="tuned_forest_model",
        ):
            train_and_save_models("dummy_train_path", "dummy_model_path")
            self.assertEqual(mock_joblib_dump.call_count, 4)


if __name__ == "__main__":
    unittest.main()
