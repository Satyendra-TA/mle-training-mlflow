import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# set the remote session to send the logs to
remote_server_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(remote_server_uri)

experiment_name = "ElasticNet_Wine"
mlflow.set_experiment(experiment_name)


def load_data(data_path):
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_X = train.drop(["quality"], axis=1)
    test_X = test.drop(["quality"], axis=1)
    train_y = train[["quality"]].copy()
    test_y = test[["quality"]].copy()
    return train_X, train_y, test_X, test_y


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


if __name__ == "__main__":
    np.random.seed(42)

    datafile_path = "../data/raw/wine-quality.csv"
    X_train, y_train, X_test, y_test = load_data(datafile_path)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5

    with mlflow.start_run():
        net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        net.fit(X_train, y_train)

        preds = net.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, preds)

        print("Elastic net model alpha = %.2f, l1 = %.2f" % (alpha, l1_ratio))
        print("Scores: rmse = %.4f, mae = %.4f, r2 = %.4f" % (rmse, mae, r2))

        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_param(key="l1_ratio", value=l1_ratio)
        mlflow.log_metric(key="rmse", value=rmse)
        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.log_artifact(datafile_path)

        print("save to {}".format(mlflow.get_artifact_uri()))

        mlflow.sklearn.log_model(net, "ElasticNetModel")
