import pandas as pd
from sklearn import linear_model

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])

reg = linear_model.LinearRegression()
reg.fit(X, y1)

reg2 = linear_model.LinearRegression()
reg2.fit(X, y2)

sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
sk_preds["Y1"] = reg.predict(X_hat)
sk_preds["Y2"] = reg2.predict(X_hat)
print(sk_preds.head())

sk_preds.to_csv("predictions/preds_linreg.csv", index=False)