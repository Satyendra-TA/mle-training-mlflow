import logging
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = "/mnt/c/Users/satyendra.mishra/Work/mle-training/"

logger = logging.getLogger(__name__)

def load_model(model_path):
    '''
    loads the model from given path

    Parameters
    ----------
    model_path : str
        path to the model

    Returns
    -------
    model : scikit learn model
        scikit learn model from saved pkl file

    '''
    with open(model_path, mode="rb") as f:
        model = pickle.load(f)
    return model

def load_test_data(test_path, target_col = None):
    '''
    loads the test data from given path

    Parameters
    ----------
    test_path : str
        path to the test data
    target_col : str, optional
        label column of the dataset

    Returns
    -------
    X_test : pd.DataFrame
        test dataframe of features
    y_test : pd.Series
        label colum of the test data

    '''
    df = pd.read_csv(test_path)
    X_test = df.drop(target_col, axis="columns")
    y_test = df[target_col].copy()
    return X_test, y_test

def evaluate_model(model, X, y):
    '''
    evaluates the given model on the provided data

    Parameters
    ----------
    model : scikit learn model
        model to evaluate
    X : pd.DataFrame
        data to evalaute the model on
    y : od.Series
        Actual label of the test data

    Returns
    -------
    mae : float
        mean absolute error
    rmse: float
        root mean square error

    '''
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return mae, rmse


if __name__ == "__main__":

    DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    DEFAULT_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("-d", "--test-data-path", type=str, default=DEFAULT_TEST_PATH)

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError, "No such file or directory"

    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError, "No such file or directory"

    model = load_model(args.model_path)
    logger.info("Model %s loaded from path %s", model.__class__.__name__, args.model_path)
    X_test, y_test = load_test_data(args.test_data_path, target_col="median_house_value")
    logger.info("Test data for evaluation: %s", args.test_data_path)
    mae, rmse = evaluate_model(model, X_test, y_test)
    logger.info("Evaluation done: mae = %.3f, mse = %.3f", mae, rmse)
