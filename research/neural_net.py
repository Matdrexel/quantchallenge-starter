import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import neural_network
from sklearn import multioutput

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y = train_data[["Y1", "Y2"]]
X = train_data.drop(columns=["Y1","Y2"])

layers = [(16, 8, 4, 2), (16, 16), (16), (8, 8, 8, 8, 8, 8), (16, 2, 16)]

best_score = -np.inf
best_layers = 0
best_model = None
for layer in layers:
    model = neural_network.MLPRegressor(hidden_layer_sizes=layer, max_iter=1000, random_state=138797)
    model = multioutput.MultiOutputRegressor(model)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=91234)
    scores = model_selection.cross_val_score(model, X, y, cv=kf, scoring="r2")
    print(scores)
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_layers = layer

print(f"The best lambda for Y was {best_layers} with an R^2 of {best_score}")

best_model.fit(X, y)

sk_preds = test_data[["id"]]
X_hat = test_data.drop(columns=["id"])
sk_preds[["Y1", "Y2"]] = best_model.predict(X_hat)
print(sk_preds.head())

sk_preds.to_csv("predictions/preds_nn.csv", index=False)