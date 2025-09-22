import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import kernel_approximation
from sklearn import pipeline

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])
sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
params = { "rbf__gamma": [0.01, 0.1, 1, 10, 100] }

pipeline1 = pipeline.Pipeline([("rbf", kernel_approximation.RBFSampler(n_components=500, random_state=2)), ("lr", linear_model.LinearRegression())])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=421)
grid1 = model_selection.GridSearchCV(pipeline1, params, scoring="r2", cv=kf)
grid1.fit(X, y1)

print(f"Best Score was {grid1.best_score_} with gamma = {grid1.best_params_["rbf__gamma"]}")
model1 = grid1.best_estimator_


pipeline2 = pipeline.Pipeline([("rbf", kernel_approximation.RBFSampler(n_components=500, random_state=3)), ("lr", linear_model.LinearRegression())])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=422)
grid2 = model_selection.GridSearchCV(pipeline2, params, scoring="r2", cv=kf)
grid2.fit(X, y2)

print(f"Best Score was {grid2.best_score_} with gamma = {grid2.best_params_["rbf__gamma"]}")
model2 = grid2.best_estimator_

sk_preds["Y1"] = model1.predict(X_hat)
sk_preds["Y2"] = model2.predict(X_hat)

sk_preds.to_csv("predictions/preds_grbf.csv", index=False)