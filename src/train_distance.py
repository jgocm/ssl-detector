import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit

# Linear Regression and Lasso
from sklearn import linear_model

# Ridge
from sklearn.linear_model import Ridge

# ElasticNet
from sklearn.linear_model import ElasticNet

# SVR
from sklearn.svm import SVR

# MLP Regressor
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('/home/ssl/Documents/ssl-detector/data/object_localization/training/log.csv')
df["BALL X ROB"] = ((df["BALL X"] - df["ROBOT X"]) * np.cos(df["ROBOT THETA"])) + ((df["BALL Y"] - df["ROBOT Y"]) * np.sin(df["ROBOT THETA"]))
df["BALL Y ROB"] = (-1 * (df["BALL X"] - df["ROBOT X"]) * np.sin(df["ROBOT THETA"])) + ((df["BALL Y"] - df["ROBOT Y"]) * np.cos(df["ROBOT THETA"]))
df["BALL ROB R POLAR"] = ((df["BALL X ROB"]  ** 2) + (df["BALL Y ROB"]  ** 2)) ** .5
df["BALL ROB TETHA POLAR"] = np.arctan(df["BALL Y ROB"] / df["BALL X ROB"])
df["CENTER_X"] = (df["X_MAX"] + df["X_MIN"]) / 2
df["CENTER_Y"] = (df["Y_MAX"] + df["Y_MIN"]) / 2
df["WIDTH"] = (df["X_MAX"] - df["X_MIN"])
df["HEIGHT"] = (df["Y_MAX"] - df["Y_MIN"])

df_test = pd.read_csv('/home/ssl/Documents/ssl-detector/data/object_localization/test/log.csv')
df_test = df_test[df_test["VALID"] == 1]
df_test.reset_index(drop=True, inplace=True)
df_test["BALL X ROB"] = ((df_test["BALL X"] - df_test["ROBOT X"]) * np.cos(df_test["ROBOT THETA"])) + ((df_test["BALL Y"] - df_test["ROBOT Y"]) * np.sin(df_test["ROBOT THETA"]))
df_test["BALL Y ROB"] = (-1 * (df_test["BALL X"] - df_test["ROBOT X"]) * np.sin(df_test["ROBOT THETA"])) + ((df_test["BALL Y"] - df_test["ROBOT Y"]) * np.cos(df_test["ROBOT THETA"]))
df_test["BALL ROB R POLAR"] = ((df_test["BALL X ROB"]  ** 2) + (df_test["BALL Y ROB"]  ** 2)) ** .5
df_test["BALL ROB TETHA POLAR"] = np.arctan(df_test["BALL Y ROB"] / df_test["BALL X ROB"])
df_test["CENTER_X"] = (df_test["X_MAX"] + df_test["X_MIN"]) / 2
df_test["CENTER_Y"] = (df_test["Y_MAX"] + df_test["Y_MIN"]) / 2
df_test["WIDTH"] = (df_test["X_MAX"] - df_test["X_MIN"])
df_test["HEIGHT"] = (df_test["Y_MAX"] - df_test["Y_MIN"])

input_data = df.loc[:,["X_MIN", "X_MAX", "Y_MIN", "Y_MAX"]].copy()
# input_data = df.loc[:,["CENTER_X", "CENTER_Y", "WIDTH", "HEIGHT"]].copy()

output_data = df.loc[:,["BALL X ROB", "BALL Y ROB"]].copy()
# output_data = df.loc[:,["BALL ROB R POLAR", "BALL ROB TETHA POLAR"]].copy()

test_input_data = df_test.loc[:,["X_MIN", "X_MAX", "Y_MIN", "Y_MAX"]].copy()
test_output_data = df_test.loc[:,["BALL X ROB", "BALL Y ROB"]].copy()

# input_data_aug = input_data.copy()
# output_data_aug = output_data.copy()

# data_size = input_data.shape[0]
# count = 0
# for idx in range(data_size):
#     for inc in range(1, 15):
#         next_point = idx + inc
#         # next_point = random.randint(1, (data_size - 1))
#         if next_point >= data_size:
#             continue
        
#         new_input = (input_data.loc[idx, :].copy() + input_data.loc[next_point, :].copy()) / 2
#         temp_df = pd.DataFrame([new_input], columns=input_data.columns, index=[data_size + count])
#         input_data_aug = pd.concat([input_data_aug, temp_df])
        
#         new_output = (output_data.loc[idx, :].copy() + output_data.loc[next_point, :].copy()) / 2
#         temp_df = pd.DataFrame([new_output], columns=output_data.columns, index=[data_size + count])
#         output_data_aug = pd.concat([output_data_aug, temp_df])

#         count += 1

# X_train, X_temp_test, y_train, y_temp_test = train_test_split(input_data_aug, output_data_aug, test_size=0.3, random_state=1)
# X_test, X_valid, y_test, y_valid = train_test_split(X_temp_test, y_temp_test, test_size=0.5, random_state=1)

# X_train, X_test, y_train, y_test = train_test_split(input_data_aug, output_data_aug, test_size=0.1, random_state=1)
# X_valid, y_valid = (X_test,y_train)

X_train, y_train = (input_data, output_data)
X_test, y_test = (test_input_data, test_output_data)
X_valid, y_valid = (X_test, y_train)

input_min_max_scaler = MinMaxScaler()
X_train_preprocessed = input_min_max_scaler.fit_transform(X_train)
X_valid_preprocessed = input_min_max_scaler.transform(X_valid)
X_test_preprocessed = input_min_max_scaler.transform(X_test)

output_min_max_scaler = MinMaxScaler()
y_train_preprocessed = output_min_max_scaler.fit_transform(y_train)
y_valid_preprocessed = output_min_max_scaler.transform(y_valid)
y_test_preprocessed = output_min_max_scaler.transform(y_test)

plt.scatter(y_train["BALL X ROB"], y_train["BALL Y ROB"])
# plt.scatter(y_test["BALL X ROB"], y_test["BALL Y ROB"])
ax = plt.gca()
ax.set_xlim([0, 3350])
ax.set_ylim([-1350, 1350])
# plt.title("Train and Test sets")
plt.show()

# plt.scatter(y_train["BALL ROB R POLAR"], y_train["BALL ROB TETHA POLAR"])
# plt.scatter(y_valid["BALL ROB R POLAR"], y_valid["BALL ROB TETHA POLAR"])
# # plt.scatter(y_test["BALL ROB R POLAR"], y_test["BALL ROB TETHA POLAR"])
# plt.show()

plt.scatter(y_train_preprocessed[:, 0], y_train_preprocessed[:, 1])
# plt.scatter(y_valid_preprocessed[:, 0], y_valid_preprocessed[:, 1])
plt.scatter(y_test_preprocessed[:, 0], y_test_preprocessed[:, 1])
plt.show()

def atomic_benchmark_estimator(estimator, X_test, verbose=False, raw=True):
    """Measure runtime prediction of each instance."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    for i in range(n_instances):
        if raw:
            instance = X_test.iloc[[i]]
        else:
            instance = X_test[[i], :]
        start = time.time()
        estimator.predict(instance)
        runtimes[i] = time.time() - start
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose):
    """Measure runtime prediction of the whole input."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_bulk_repeats, dtype=float)
    for i in range(n_bulk_repeats):
        start = time.time()
        estimator.predict(X_test)
        runtimes[i] = time.time() - start
    runtimes = np.array(list(map(lambda x: x / float(n_instances), runtimes)))
    if verbose:
        print(
            "bulk_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def benchmark_estimator(estimator, X_test, n_bulk_repeats=30, verbose=True, raw=True):
    """
    Measure runtimes of prediction in both atomic and bulk mode.

    Parameters
    ----------
    estimator : already trained estimator supporting `predict()`
    X_test : test input
    n_bulk_repeats : how many times to repeat when evaluating bulk mode

    Returns
    -------
    atomic_runtimes, bulk_runtimes : a pair of `np.array` which contain the
    runtimes in seconds.

    """
    atomic_runtimes = atomic_benchmark_estimator(estimator, X_test, verbose, raw)
    bulk_runtimes = bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose)
    return atomic_runtimes, bulk_runtimes

def train_and_test_in(model, xtrain, ytrain, xvalid, yvalid, xtest, ytest, model_name, parameters):
    # Setting GridSearch 
    # parameters = {
    #     'hidden_layer_sizes': [
    #         (10,10,),
    #         (30,30,),
    #         (50,50,),
    #         (100,100,),
    #         (10,10,10,),
    #         (30,30,30,),
    #         (50,50,50,),
    #         (100,100,100,),
    #         (10,10,10,10,10,),
    #         (30,30,30,30,30,),
    #         (50,50,50,50,50,),
    #         (100,100,100,100,100,)
    #     ],
    #     'learning_rate_init': [0.01, 0.001],
    #     'random_state': [3],
    #     'tol': [1e-4, 1e-5, 1e-6, 1e-7],
    #     'max_iter': [50000]
    # }

    # Training
    # X_concat = np.concatenate((xtrain, xvalid))
    # y_concat = np.concatenate((ytrain, yvalid))
    # train_split = np.zeros(len(xtrain)) - 1
    # test_split = np.ones(len(xvalid))
    # ps = PredefinedSplit(np.concatenate((train_split, test_split)))
    # reg_search = GridSearchCV(model, param_grid=parameters, cv=ps,  return_train_score=True, scoring='neg_mean_squared_error')
    # reg_search.fit(X_concat, y_concat)
    reg_search = GridSearchCV(model, param_grid=parameters, cv=5,  return_train_score=True, scoring='neg_mean_squared_error', verbose=3)
    reg_search.fit(xtrain, ytrain)

    # Evaluating
    print(f"Bes tParameters {reg_search.best_params_}")
    y_test_pred = reg_search.predict(xtest)

    processed = "Preprocessed Input" in model_name
    raw = not processed
    print (raw)
    atomic_runtimes, bulk_runtimes = benchmark_estimator(reg_search, xtest, 30, True, raw)
    print(f"Atomic Runtime {atomic_runtimes}")
    print(f"Bulk Runtime {bulk_runtimes}")
    
    if "Preprocessed Input and Output" in model_name:
        y_test_pred = output_min_max_scaler.inverse_transform(y_test_pred)
        ytest = output_min_max_scaler.inverse_transform(ytest)
        scatter_gt = plt.scatter(ytest[:, 0], ytest[:, 1])
    else:
        scatter_gt = plt.scatter(ytest["BALL X ROB"], ytest["BALL Y ROB"])
        # plt.scatter(ytest["BALL ROB R POLAR"], ytest["BALL ROB TETHA POLAR"])

    print (f"RMSE: {mean_squared_error(ytest, y_test_pred, squared=False)}")

    scatter_pred = plt.scatter(y_test_pred[:, 0], y_test_pred[:, 1])
    # plt.text(400, 1100, f"RMSE: {mean_squared_error(ytest, y_test_pred, squared=False)}")
    # plt.title(model_name)
    plt.legend((scatter_gt, scatter_pred),
           ('Ground truth Positions', 'Predicted Positions'),
           loc='upper right',
           fontsize=12)
    ax = plt.gca()
    ax.set_xlim([0, 3350])
    ax.set_ylim([-1350, 1350])
    plt.show()
    
    # plt.scatter(ytest["BALL ROB R POLAR"] * np.cos(ytest["BALL ROB TETHA POLAR"]), ytest["BALL ROB R POLAR"] * np.sin(ytest["BALL ROB TETHA POLAR"]))
    # plt.scatter(y_test_pred[:, 0] * np.cos(y_test_pred[:, 1]), y_test_pred[:, 0] * np.sin(y_test_pred[:, 1]))
    # plt.title(model_name)
    # plt.show()
    
    
def train_and_test(model, model_name, parameters):
    train_and_test_in(model, X_train, y_train, X_valid, y_valid, X_test, y_test, model_name + " - Raw", parameters)

def train_and_test_preprocessed_input(model, model_name, parameters):
    train_and_test_in(model, X_train_preprocessed, y_train, X_valid_preprocessed, y_valid, X_test_preprocessed, y_test, model_name + " - Preprocessed Input", parameters)
    
def train_and_test_preprocessed_all(model, model_name, parameters):
    train_and_test_in(model, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, model_name + " - Preprocessed Input and Output", parameters)

mlp_reg = MLPRegressor()
parameters = {
    'hidden_layer_sizes': [
        (10,10,)
    ],
    'learning_rate_init': [0.01, 0.001],
    'random_state': [3],
    'tol': [1e-4],
    'max_iter': [50000]
}
train_and_test(mlp_reg, "MLP", parameters)

# mlp_reg = MLPRegressor(random_state=1, max_iter=10000, hidden_layer_sizes = (100,100,100,), learning_rate_init=0.01)
# train_and_test(mlp_reg, "MLP")

mlp_reg = MLPRegressor()
parameters = {
    'hidden_layer_sizes': [
        (50, 50, 50, 50, 50,)
    ],
    'learning_rate_init': [0.001,0.01],
    'random_state': [3],
    'tol': [1e-4],
    'max_iter': [50000]
}
train_and_test_preprocessed_input(mlp_reg, "MLP", parameters)

# mlp_reg = MLPRegressor(random_state=3, max_iter=10000, hidden_layer_sizes = (100,100,100,), learning_rate_init=0.01)
# train_and_test_preprocessed_input(mlp_reg, "MLP")

mlp_reg = MLPRegressor()
# parameters = {
#     'hidden_layer_sizes': [
#         (10,10,),
#         (30,30,),
#         (50,50,),
#         (100,100,),
#         (10,10,10,),
#         (30,30,30,),
#         (50,50,50,),
#         (100,100,100,),
#         (10,10,10,10,10,),
#         (30,30,30,30,30,),
#         (50,50,50,50,50,),
#         (100,100,100,100,100,)
#     ],
#     'learning_rate_init': [0.01, 0.001, 0.0001],
#     'random_state': [3],
#     'tol': [1e-7, 1e-8],
#     'max_iter': [50000]
# }
parameters = {
    'hidden_layer_sizes': [
        (100, 100, 100, 100, 100)
    ],
    'learning_rate_init': [0.01, 0.001],
    'random_state': [3],
    'tol': [1e-7],
    'max_iter': [50000]
}
train_and_test_preprocessed_all(mlp_reg, "MLP", parameters)

# mlp_reg = MLPRegressor(random_state=1, max_iter=10000, hidden_layer_sizes = (100,100,100,), learning_rate_init=0.01)
# train_and_test_preprocessed_all(mlp_reg, "MLP")