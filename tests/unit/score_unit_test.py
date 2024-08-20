import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.components.score import score_models


class TestScoreModule(unittest.TestCase):

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
                "ocean_proximity": ["NEAR BAY", "INLAND"],
                "median_house_value": [452600, 358500],
            }
        )

        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([500000, 350000])

    @patch("pandas.read_csv")
    @patch("joblib.load")
    @patch("os.path.isdir", return_value=True)
    @patch("os.listdir", return_value=["model.pkl"])
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_score_models(
        self,
        mock_to_csv,
        mock_makedirs,
        mock_listdir,
        mock_isdir,
        mock_load,
        mock_read_csv,
    ):
        mock_read_csv.return_value = self.sample_data
        mock_load.return_value = self.mock_model

        score_models("dummy_model_path", "dummy_data_path",
                     "dummy_output_path")

        # Check that load was called as many times as there are model files
        expected_calls = len(mock_listdir.return_value)
        self.assertEqual(mock_load.call_count, expected_calls)
        mock_makedirs.assert_called_once_with("dummy_output_path")
        mock_load.assert_called_with("dummy_model_path/model.pkl")
        mock_to_csv.assert_called()
        self.mock_model.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
