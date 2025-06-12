## Hyperparameter optimization for XGboost


## rapip install -r requirements.txt

import os
from urllib.request import urlretrieve

import cudf
import dask_ml.model_selection as dcv
import numpy as np
import pandas as pd
import xgboost as xgb
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from sklearn.metrics import make_scorer



print("cudf version: ",cudf.__version__)

## Setting up client with dask

cluster = LocalCUDACluster()
client = Client(cluster)
client

## Prepare dataset
data_dir = "./rapids_hpo/data/"
file_name = "airlines.parquet"
parquet_name = os.path.join(data_dir, file_name)

parquet_name

def prepare_dataset(use_full_dataset=False):
    global file_path, data_dir

    if use_full_dataset:
        url = "https://data.rapids.ai/cloud-ml/airline_20000000.parquet"
    else:
        url = "https://data.rapids.ai/cloud-ml/airline_small.parquet"

    if os.path.isfile(parquet_name):
        print(f" > File already exists. Ready to load at {parquet_name}")
    else:
        # Ensure folder exists
        os.makedirs(data_dir, exist_ok=True)

        def data_progress_hook(block_number, read_size, total_filesize):
            if (block_number % 1000) == 0:
                print(
                    f" > percent complete: { 100 * ( block_number * read_size ) / total_filesize:.2f}\r",
                    end="",
                )
            return

        urlretrieve(
            url=url,
            filename=parquet_name,
            reporthook=data_progress_hook,
        )

        print(f" > Download complete {file_name}")

    input_cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
    ]

    dataset = cudf.read_parquet(parquet_name)

    # encode categoricals as numeric
    for col in dataset.select_dtypes(["object"]).columns:
        dataset[col] = dataset[col].astype("category").cat.codes.astype(np.int32)

    # cast all columns to int32
    for col in dataset.columns:
        dataset[col] = dataset[col].astype(np.float32)  # needed for random forest

    # put target/label column first [ classic XGBoost standard ]
    output_cols = ["ArrDelayBinary"] + input_cols

    dataset = dataset.reindex(columns=output_cols)
    return dataset


df = prepare_dataset()

import time
from contextlib import contextmanager

# Helping time blocks of code
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"{txt:>32} time:  {t1-t0:8.5f}")


# Define some default values to make use of across the notebook for a fair comparison
N_FOLDS = 5
N_ITER = 25

label = "ArrDelayBinary"

# Split data

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.2)
X_cpu = X_train.to_pandas()
y_cpu = y_train.to_numpy()

X_test_cpu = X_test.to_pandas()
y_test_cpu = y_test.to_numpy()


def accuracy_score_wrapper(y, y_hat):
    """
    A wrapper function to convert labels to float32,
    and pass it to accuracy_score.

    Params:
    - y: The y labels that need to be converted
    - y_hat: The predictions made by the model
    """
    y = y.astype("float32")  # cuML RandomForest needs the y labels to be float32
    return accuracy_score(y, y_hat, convert_dtype=True)


accuracy_wrapper_scorer = make_scorer(accuracy_score_wrapper)
cuml_accuracy_scorer = make_scorer(accuracy_score, convert_dtype=True)


def do_HPO(model, gridsearch_params, scorer, X, y, mode="gpu-Grid", n_iter=10):
    """
    Perform HPO based on the mode specified

    mode: default gpu-Grid. The possible options are:
    1. gpu-grid: Perform GPU based GridSearchCV
    2. gpu-random: Perform GPU based RandomizedSearchCV

    n_iter: specified with Random option for number of parameter settings sampled

    Returns the best estimator and the results of the search
    """
    if mode == "gpu-grid":
        print("gpu-grid selected")
        clf = dcv.GridSearchCV(model, gridsearch_params, cv=N_FOLDS, scoring=scorer)
    elif mode == "gpu-random":
        print("gpu-random selected")
        clf = dcv.RandomizedSearchCV(
            model, gridsearch_params, cv=N_FOLDS, scoring=scorer, n_iter=n_iter
        )

    else:
        print("Unknown Option, please choose one of [gpu-grid, gpu-random]")
        return None, None
    res = clf.fit(X, y)
    print(f"Best clf and score {res.best_estimator_} {res.best_score_}\n---\n")
    return res.best_estimator_, res


def print_acc(model, X_train, y_train, X_test, y_test, mode_str="Default"):
    """
    Trains a model on the train data provided, and prints the accuracy of the trained model.
    mode_str: User specifies what model it is to print the value
    """
    y_pred = model.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test.astype("float32"), convert_dtype=True)
    print(f"{mode_str} model accuracy: {score}")



X_train.shape

# Default performance -- we first use the model with it's default parameter and see the accuracy

model_gpu_xgb_ = xgb.XGBClassifier(tree_method="gpu_hist")

print_acc(model_gpu_xgb_, X_train, y_cpu, X_test, y_test_cpu)

# Parameter Distributions

# For xgb_model
model_gpu_xgb = xgb.XGBClassifier(tree_method="gpu_hist")

# More range
params_xgb = {
    "max_depth": np.arange(start=3, stop=12, step=3),  # Default = 6
    "alpha": np.logspace(-3, -1, 5),  # default = 0
    "learning_rate": [0.05, 0.1, 0.15],  # default = 0.3
    "min_child_weight": np.arange(start=2, stop=10, step=3),  # default = 1
    "n_estimators": [100, 200, 1000],
}


# Randomized Search CV

mode = "gpu-random"

with timed("XGB-" + mode):
    res, results = do_HPO(
        model_gpu_xgb,
        params_xgb,
        cuml_accuracy_scorer,
        X_train,
        y_cpu,
        mode=mode,
        n_iter=N_ITER,
    )
num_params = len(results.cv_results_["mean_test_score"])
print(f"Searched over {num_params} parameters")

print_acc(res, X_train, y_cpu, X_test, y_test_cpu, mode_str=mode)

mode = "gpu-grid"

with timed("XGB-" + mode):
    res, results = do_HPO(
        model_gpu_xgb, params_xgb, cuml_accuracy_scorer, X_train, y_cpu, mode=mode
    )
num_params = len(results.cv_results_["mean_test_score"])
print(f"Searched over {num_params} parameters")

print_acc(res, X_train, y_cpu, X_test, y_test_cpu, mode_str=mode)
