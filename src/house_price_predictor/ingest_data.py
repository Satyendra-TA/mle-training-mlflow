import os
import tarfile
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
    _fit_transform_one,
    _transform_one,
)
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
    """Adds attributes to the input dataset

    Parameters
    ----------
    add_bedroom_per_room : bool
        whether to add bedrooms per room
    idx : list[int]
        indexes of columns for feature engineering

    Methods
    -------
    fit(X, y=None)
        fits the model on the dataset X
    transform(X)
        transforms X based on statistics learned in fit()

    """

    def __init__(self, idx, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.idx = idx

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        columns = X.columns.tolist()
        X = X.to_numpy()
        rooms_ix, bedrooms_ix, population_ix, households_ix = self.idx
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            columns += [
                "rooms_per_household",
                "population_per_household",
                "bedrooms_per_room",
            ]
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            modified_df = pd.DataFrame(
                np.c_[
                    X,
                    rooms_per_household,
                    population_per_household,
                    bedrooms_per_room,
                ],
                columns=columns,
            )
            return modified_df
        else:
            columns += ["rooms_per_household", "population_per_household"]
            modified_df = pd.DataFrame(
                np.c_[
                    X,
                    rooms_per_household,
                    population_per_household,
                ],
                columns=columns,
            )
            return modified_df


class CustomSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, strategy):
        self.variables = variables
        self.strategy = strategy
        self.imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)

    def fit(self, X, y=None):
        X_ = X.loc[:, self.variables]
        self.imp.fit(X_)
        return self

    def transform(self, X):
        X_ = X.loc[:, self.variables]
        X_transformed = pd.DataFrame(
            self.imp.transform(X_), columns=self.variables
        )
        X.drop(self.variables, axis=1, inplace=True)
        X[self.variables] = X_transformed[self.variables].values
        return X


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_ = X.loc[:, self.variables]
        self.scaler.fit(X_)
        return self

    def transform(self, X):
        X_ = X.loc[:, self.variables]
        # get one-hot encoded feature in df format
        X_transformed = pd.DataFrame(
            self.scaler.transform(X_),
            columns=self.scaler.get_feature_names_out(),
        )

        # Remove columns that are one hot encoded in original df
        X.drop(self.variables, axis=1, inplace=True)

        # Add one hot encoded feature to original df
        X[self.scaler.get_feature_names_out()] = X_transformed[
            self.scaler.get_feature_names_out()
        ].values
        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables, sparse_output=True):
        self.variables = variables
        self.sparse_output = sparse_output
        self.ohe = OneHotEncoder(
            drop="first",
            sparse_output=self.sparse_output,
            handle_unknown="ignore",
        )

    def fit(self, X, y=None):
        X_ = X.loc[:, self.variables]
        self.ohe.fit(X_)
        return self

    def transform(self, X):
        X_ = X.loc[:, self.variables]
        # get one-hot encoded feature in df format
        array = (
            self.ohe.transform(X_).toarray()
            if self.sparse_output
            else self.ohe.transform(X_)
        )
        X_transformed = pd.DataFrame(
            array,
            columns=self.ohe.get_feature_names_out(),
        )

        # Remove columns that are one hot encoded in original df
        X.drop(self.variables, axis=1, inplace=True)

        # Add one hot encoded feature to original df
        X[self.ohe.get_feature_names_out()] = X_transformed[
            self.ohe.get_feature_names_out()
        ].values
        return X


class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans, X=X, y=y, weight=weight, **fit_params
            )
            for name, trans, weight in self._iter()
        )
        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        # print(Xs[1])
        self._update_transformer_list(transformers)
        if any(scipy.sparse.issparse(f) for f in Xs):
            Xs = scipy.sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        concatenated = pd.concat(Xs, axis="columns", copy=False)
        return concatenated

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans, X=X, y=None, weight=weight
            )
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(scipy.sparse.issparse(f) for f in Xs):
            Xs = scipy.sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


def get_preprocessing_pipeline(df, num_attributes, cat_attributes):
    df = df.copy()
    # column indices for feature enfinnering
    col_names = ["total_rooms", "total_bedrooms", "population", "households"]
    indices = get_column_indices(df, col_names)

    # pipeline for numerical attributes
    num_pipeline = Pipeline(
        [
            (
                "imputer",
                CustomSimpleImputer(num_attributes, strategy="median"),
            ),
            (
                "attribs_adder",
                CombinedAttributesAdder(indices, add_bedrooms_per_room=False),
            ),
            ("std_scaler", CustomStandardScaler(num_attributes)),
        ]
    )

    # pipeline for categorical variables
    cat_pipeline = Pipeline([("OHE", CustomOneHotEncoder(cat_attributes))])

    # full pipeline
    full_pipeline = PandasFeatureUnion(
        [("numerical_pipe", num_pipeline), ("categorical_pipe", cat_pipeline)]
    )

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
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
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
    target = ["median_house_value"]
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

    # preprocess the test set
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_prepared = preprocessing_pipeline.transform(X_test)

    # save the feature and label files as csv
    housing_prepared.to_csv(
        os.path.join(args.output_path, "train_X.csv"), index=None
    )
    housing_labels.to_csv(
        os.path.join(args.output_path, "train_labels.csv"), index=None
    )
    X_test_prepared.to_csv(
        os.path.join(args.output_path, "test_X.csv"), index=None
    )
    y_test.to_csv(
        os.path.join(args.output_path, "test_labels.csv"), index=None
    )
