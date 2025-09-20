import pandas as pd
import numpy as np

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])

w1 = np.linalg.solve(X.T @ X, X.T @ y1)
w2 = np.linalg.solve(X.T @ X, X.T @ y2)

preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
preds["Y1"] = X_hat @ w1
preds["Y2"] = X_hat @ w2
print(preds.head())

preds.to_csv("preds_linreg.csv", index=False)