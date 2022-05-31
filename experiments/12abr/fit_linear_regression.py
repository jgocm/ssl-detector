from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os

cwd = os.getcwd()

bboxes = pd.read_csv(cwd+f"/experiments/12abr/bounding_boxes.txt", sep=" ", header=None)
positions = pd.read_csv(cwd+f"/experiments/12abr/field_pixels.txt", sep=" ", header=None)

# X AXIS LINEAR REGRESSION
X = bboxes.to_numpy()
y = positions.iloc[:,:-1].to_numpy().T[0]

kf = KFold(n_splits=5, random_state=None, shuffle=False)

models = [linear_model.LinearRegression(), linear_model.Ridge(alpha=0.5), linear_model.BayesianRidge()]

print("Average precisions for X axis:")
for model in models:
    avg_score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        avg_score+=model.score(X_test, y_test)
    avg_score = avg_score/5
    print(avg_score)

# Y AXIS LINEAR REGRESSION
X = bboxes.to_numpy()
y = positions.iloc[:,-1:].to_numpy().T[0]

kf = KFold(n_splits=5, random_state=None, shuffle=False)

models = [linear_model.LinearRegression(), linear_model.Ridge(alpha=0.5), linear_model.BayesianRidge()]

print("Average precisions for Y axis:")
for model in models:
    avg_score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        avg_score+=model.score(X_test, y_test)
    avg_score = avg_score/5
    print(avg_score)

# NO DIFFERENCE BETWEEN MODELS => USE THE LIGHTEST ONE
X = bboxes
y = positions

model = linear_model.LinearRegression().fit(X,y)
coefficients = model.coef_
intercept = model.intercept_

x_coef = np.append(coefficients[0],intercept[0])
print(x_coef)
y_coef = np.append(coefficients[1],intercept[1])
print(y_coef)
np.savetxt(cwd+f"/experiments/12abr/regression_weights.txt", [x_coef, y_coef])