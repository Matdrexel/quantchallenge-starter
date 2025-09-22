import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import model_selection

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])

model1 = linear_model.LinearRegression()
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=12378)
feature_select1 = feature_selection.SequentialFeatureSelector(model1, direction="forward", scoring="r2", cv=kf)

feature_select1.fit(X, y1)
selected_features1 = X.columns[feature_select1.get_support()]

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=1278)
scores1 = model_selection.cross_val_score(model1, X[selected_features1], y1, cv=kf, scoring="r2")
print(f"R2 score for Y1 with columns {selected_features1} was {np.mean(scores1)}")

model2 = linear_model.LinearRegression()
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=1237)
feature_select2 = feature_selection.SequentialFeatureSelector(model2, direction="forward", scoring="r2", cv=kf)

feature_select2.fit(X, y2)
selected_features2 = X.columns[feature_select2.get_support()]

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=1378)
scores2 = model_selection.cross_val_score(model2, X[selected_features2], y2, cv=kf, scoring="r2")
print(f"R2 score for Y2 with columns {selected_features2} was {np.mean(scores2)}")
print(X[selected_features2])

model1.fit(X[selected_features1], y1)
model2.fit(X[selected_features2], y2)

sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
sk_preds["Y1"] = model1.predict(X_hat[selected_features1])
sk_preds["Y2"] = model2.predict(X_hat[selected_features2])
print(sk_preds.head())

sk_preds.to_csv("predictions/preds_linreg_fs.csv", index=False)