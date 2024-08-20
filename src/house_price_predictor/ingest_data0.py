import os
import tarfile
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
PROJECT_ROOT = "/mnt/c/Users/satyendra.mishra/Work/mle-training/"
HOUSING_PATH = os.path.join(PROJECT_ROOT, "data", "raw")


def fetch_housing_data(
    housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH
) -> None:
    """
    Fetches the housing price datasey from the provided url

    Parameters
    ----------
    housing_url : str, optional
        source URL of the housing price dataset
    housing_path : str, optional
        local directory to store the housing dataset

    Returns
    -------
    None

    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    """
    loads the housing data into a pandas dataframe

    Parameters
    ----------
    housing_path : str
        local path to the housing data

    Returns
    -------
        a pandas datarame of housing data

    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def get_column_indices(df, col_names):
    """
    finds the indices of column names in the dataframe's list of columns

    Parameter
    ---------
    df : pd.DataFrame
        input dataframe
    col_names: list[str]
        list of column names whoswe indices are to be determined

    """
    return [df.columns.get_loc(c) for c in col_names]


class CombinedAttributesAdder(
    BaseEstimator, TransformerMixin, auto_wrap_output_keys=None
):
    """Adds attributes to the input dataset"""

    def __init__(self, idx, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.idx = idx

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = self.idx
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    # def get_feature_names_out(self, *args, **kwargs):
    #     if self.add_bedrooms_per_room:
    #         return self.columns_ + [
    #             "rooms_per_household",
    #             "population_per_household",
    #             "bedrooms_per_room",
    #         ]
    #     else:
    #         return self.columns_ + [
    #             "rooms_per_household",
    #             "population_per_household",
    #         ]


def get_preprocessing_pipeline(
    df, num_attributes, cat_attributes, output_mode="pandas"
):
    # column indices for feature enfinnering
    col_names = ["total_rooms", "total_bedrooms", "population", "households"]
    indices = get_column_indices(df, col_names)
    # pipeline for numerical attributes
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder(indices)),
            ("std_scaler", StandardScaler()),
        ]
    )
    # complete preprocessing pipeline
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attributes),
            ("cat", OneHotEncoder(), cat_attributes),
        ],
        sparse_threshold=0,
    )

    # full_pipeline.set_output(transform=output_mode)
    return full_pipeline


def split_data(dataset, test_size):
    """
    splits the dataset into training and testing data

    Parameters
    ----------
    dataset : pd.DataFrame
        dataset to split
    test_size : float
        size of the test set as a fraction to total dataset size

    Returns
    -------
    train : pd.DataFrame
        training dataset

    test : pd.DataFrame
        test dataset

    """
    train, test = train_test_split(
        dataset, test_size=test_size, random_state=42
    )
    return train, test


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HOUSING_PATH, "housing.csv")):
        fetch_housing_data()

    desc = """Fetches data, performs transformations,
              splits into train and validation sets
              and saves the splits in specified directory"""

    DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    parser = ArgumentParser(prog="ingest-data", description=desc)
    parser.add_argument("-o", "--output-path", type=str, default=DEFAULT_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        raise "The specified path %s does not exist" % (args.output_path)
    # load the data
    housing_data = load_housing_data()
    # train test split
    train_set, test_set = split_data(housing_data, test_size=0.2)

    # Features dataframe
    housing_df = train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set

    # training labels
    housing_labels = train_set["median_house_value"].copy()
    # numerical features
    housing_num = housing_df.drop("ocean_proximity", axis=1)

    # preprocessing pipeline
    num_attribs = housing_num.columns.tolist()
    cat_attribs = ["ocean_proximity"]

    preprocessing_pipeline = get_preprocessing_pipeline(
        housing_df, num_attribs, cat_attribs
    )
    housing_prepared = preprocessing_pipeline.fit_transform(housing_df)

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_prepared = preprocessing_pipeline.transform(X_test)

    pd.DataFrame(housing_prepared).to_csv(
        os.path.join(args.output_path, "train_X.csv"), index=None
    )
    pd.DataFrame(housing_labels).to_csv(
        os.path.join(args.output_path, "train_labels.csv"), index=None
    )
    pd.DataFrame(X_test_prepared).to_csv(
        os.path.join(args.output_path, "test_X.csv"), index=None
    )
    pd.DataFrame(y_test).to_csv(
        os.path.join(args.output_path, "test_labels.csv"), index=None
    )
