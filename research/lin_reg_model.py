import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing


train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])

pipeline1 = pipeline.Pipeline([("lr", linear_model.LinearRegression())])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=913)
scores = model_selection.cross_val_score(pipeline1, X, y1, cv=kf, scoring="r2")
print(scores, np.mean(scores))
pipeline1.fit(X, y1)

pipeline2 = pipeline.Pipeline([("lr", linear_model.LinearRegression())])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=9)
scores = model_selection.cross_val_score(pipeline2, X, y2, cv=kf, scoring="r2")
print(scores, np.mean(scores))
pipeline2.fit(X, y2)

sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
sk_preds["Y1"] = pipeline1.predict(X_hat)
sk_preds["Y2"] = pipeline2.predict(X_hat)
print(sk_preds.head())

sk_preds.to_csv("predictions/preds_linreg.csv", index=False)