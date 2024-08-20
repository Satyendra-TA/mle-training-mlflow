import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.components.ingest_data import (fetch_housing_data,
                                        load_housing_data, split_data)


class TestHousingDataModule(unittest.TestCase):
    def setUp(self):
        self.housing_url = """https://raw.githubusercontent.com/ageron/
        handson-ml/master/datasets/housing/housing.tgz"""
        self.housing_path = """/mnt/c/Users/sai.mallampally/mle-training/
        artifacts/housing/housing.tgz"""

    @patch("os.makedirs")
    @patch("urllib.request.urlretrieve")
    @patch("tarfile.open")
    def test_fetch_housing_data(
        self, mock_tarfile_open, mock_urlretrieve, mock_makedirs
    ):
        # Set up the tarfile.open mock to use the context manager correctly
        mock_tar = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        fetch_housing_data(self.housing_url, self.housing_path)

        mock_makedirs.assert_called_with(self.housing_path, exist_ok=True)
        mock_urlretrieve.assert_called_with(
            self.housing_url, os.path.join(self.housing_path, "housing.tgz")
        )
        mock_tar.extractall.assert_called_with(path=self.housing_path)

    @patch("pandas.read_csv")
    def test_load_housing_data(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame()
        result = load_housing_data(self.housing_path)
        self.assertIsInstance(result, pd.DataFrame)

    def test_split_data(self):
        data = pd.DataFrame({"feature1": range(100), "feature2": range(100,
                                                                       200)})
        train, test = split_data(data, test_size=0.2, seed=42)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)


if __name__ == "__main__":
    unittest.main()
