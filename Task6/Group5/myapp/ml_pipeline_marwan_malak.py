import numpy as np
import pickle
import os

# Paths to models inside myapp/models
models_dir = os.path.join("myapp", "models")

models_dir = os.path.join(os.path.dirname(__file__), "models")

# loading
transformer = pickle.load(
    open(os.path.join(models_dir, "transformer.pkl"), "rb")
)  # for ridge, dtree
transformer2 = pickle.load(
    open(os.path.join(models_dir, "transformer2.pkl"), "rb")
)  # for xgb, linear, knn

ridge_model = pickle.load(open(os.path.join(models_dir, "ridgeregression.pkl"), "rb"))
dtree_model = pickle.load(open(os.path.join(models_dir, "decisiontree.pkl"), "rb"))
linear_model = pickle.load(open(os.path.join(models_dir, "Linear.pkl"), "rb"))
xgb_model = pickle.load(open(os.path.join(models_dir, "XGB.pkl"), "rb"))


# prediction
def predict_fare(features):

    # XGB, Linear, KNN (trained with transformer2.pkl)
    transformed_v2 = transformer2.transform(features)
    xgb_pred = xgb_model.predict(transformed_v2)[0]
    linear_pred = linear_model.predict(transformed_v2)[0]

    return xgb_pred, linear_pred
