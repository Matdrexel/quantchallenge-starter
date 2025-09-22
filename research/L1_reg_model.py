import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection

train_data = pd.read_csv('./data/train_new.csv')
test_data = pd.read_csv('./data/test_new.csv')

y1 = train_data["Y1"]
y2 = train_data["Y2"]
X = train_data.drop(columns=["Y1","Y2"])

lammies = [((x % 10) + 1) * 10**(-3+(x // 10)) for x in range(50)]

best_score1 = -np.inf
best_lammy1 = 0
best_model1 = None
for lammy in lammies:
    model = linear_model.Lasso(alpha=lammy)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=98462)
    scores = model_selection.cross_val_score(model, X, y1, cv=kf, scoring="r2")
    mean_score = np.mean(scores)
    if mean_score > best_score1:
        best_score1 = mean_score
        best_model1 = model
        best_lammy1 = lammy

print(f"The best lambda for Y1 was {best_lammy1} with an R^2 of {best_score1}")

best_score2 = -np.inf
best_lammy2 = 0
best_model2 = None
for lammy in lammies:
    model = linear_model.Lasso(alpha=lammy)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=98463)
    scores = model_selection.cross_val_score(model, X, y2, cv=kf, scoring="r2")
    mean_score = np.mean(scores)
    if mean_score > best_score2:
        best_score2 = mean_score
        best_model2 = model
        best_lammy2 = lammy

print(f"The best lambda for Y2 was {best_lammy2} with an R^2 of {best_score2}")

best_model1.fit(X, y1)
best_model2.fit(X, y2)

sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
sk_preds["Y1"] = best_model1.predict(X_hat)
sk_preds["Y2"] = best_model2.predict(X_hat)
print(sk_preds.head())

sk_preds.to_csv("predictions/preds_linreg_L1reg.csv", index=False)