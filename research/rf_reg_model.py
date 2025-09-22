import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import ensemble

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])
sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
params = { "rf__n_estimators": [50, 100, 150, 200], "rf__max_depth": [5, 10, 15, 20] }

pipeline1 = pipeline.Pipeline([("rf", ensemble.RandomForestRegressor(random_state=1224))])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=425)
grid1 = model_selection.GridSearchCV(pipeline1, params, scoring="r2", cv=kf)
grid1.fit(X, y1)

print(f"Best Score was {grid1.best_score_} with nTrees = {grid1.best_params_["rf__n_estimators"]} and maxDepth = {grid1.best_params_["rf__max_depth"]}")
model1 = grid1.best_estimator_


pipeline2 = pipeline.Pipeline([("rf", ensemble.RandomForestRegressor(random_state=1225))])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=426)
grid2 = model_selection.GridSearchCV(pipeline2, params, scoring="r2", cv=kf)
grid2.fit(X, y2)

print(f"Best Score was {grid2.best_score_} with nTrees = {grid2.best_params_["rf__n_estimators"]} and maxDepth = {grid2.best_params_["rf__max_depth"]}")
model2 = grid2.best_estimator_

sk_preds["Y1"] = model1.predict(X_hat)
sk_preds["Y2"] = model2.predict(X_hat)

sk_preds.to_csv("predictions/preds_rf.csv", index=False)