import pandas as pd
from xgboost import XGBRegressor
from sklearn import model_selection

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])
sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
params = { "n_estimators": [50, 100, 150, 200], "max_depth": [5, 10, 15, 20], "learning_rate": [0.01, 0.05, 0.1, 0.3] }

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=426)
grid1 = model_selection.GridSearchCV(XGBRegressor(random_state=89, n_jobs=-1), params, scoring="r2", cv=kf)
grid1.fit(X, y1)

print(f"Best Score was {grid1.best_score_} with nTrees = {grid1.best_params_["n_estimators"]}, maxDepth = {grid1.best_params_["max_depth"]}, and learning_rate = {grid1.best_params_["learning_rate"]}")
model1 = grid1.best_estimator_


kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=427)
grid2 = model_selection.GridSearchCV(XGBRegressor(random_state=88, n_jobs=-1), params, scoring="r2", cv=kf)
grid2.fit(X, y2)

print(f"Best Score was {grid2.best_score_} with nTrees = {grid2.best_params_["n_estimators"]}, maxDepth = {grid2.best_params_["max_depth"]}, and learning_rate = {grid2.best_params_["learning_rate"]}")
model2 = grid2.best_estimator_

sk_preds["Y1"] = model1.predict(X_hat)
sk_preds["Y2"] = model2.predict(X_hat)

sk_preds.to_csv("predictions/preds_xgb.csv", index=False)