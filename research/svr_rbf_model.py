import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])
sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
params = { "svr__C": [0.01], "svr__gamma": ["scale"] }

pipeline1 = pipeline.Pipeline([("scale", preprocessing.StandardScaler()), ("svr", svm.SVR(kernel="rbf"))])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=423)
grid1 = model_selection.GridSearchCV(pipeline1, params, scoring="r2", cv=kf)
grid1.fit(X, y1)

print(f"Best Score was {grid1.best_score_} with gamma = {grid1.best_params_["svr__gamma"]} and C = {grid1.best_params_["svr__C"]}")
model1 = grid1.best_estimator_


pipeline2 = pipeline.Pipeline([("scale", preprocessing.StandardScaler()), ("svr", svm.SVR(kernel="rbf"))])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=424)
grid2 = model_selection.GridSearchCV(pipeline2, params, scoring="r2", cv=kf)
grid2.fit(X, y2)

print(f"Best Score was {grid2.best_score_} with gamma = {grid2.best_params_["svr__gamma"]} and C = {grid2.best_params_["svr__C"]}")
model2 = grid2.best_estimator_

sk_preds["Y1"] = model1.predict(X_hat)
sk_preds["Y2"] = model2.predict(X_hat)

sk_preds.to_csv("predictions/preds_svr_rbf.csv", index=False)