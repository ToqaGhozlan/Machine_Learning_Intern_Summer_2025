import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import pickle
from sklearn.compose import ColumnTransformer

# ---------------- Dataset Path ----------------
path = r"D:\AI_road_map\python\dataSets\final_internship_data.csv"
data = pd.read_csv(path)

# ---------------- Data Cleaning ----------------
# Remove invalid values
data = data.drop(data[data["fare_amount"] > 50][data["distance"] < 10].index, axis=0)
data = data.drop(data[data["distance"] > 50].index, axis=0)
data = data.drop(data[data["distance"] < 0.1].index, axis=0)
data = data.drop(data[data["fare_amount"] < 1].index, axis=0)

# Remove unnecessary columns
data.drop(
    labels=[
        "User ID",
        "User Name",
        "Driver Name",
        "key",
        "pickup_datetime",
        "Weather",
        "Traffic Condition",
        "Car Condition",
    ],
    axis=1,
    inplace=True,
)

# Remove null values
data.dropna(axis=0, inplace=True)

# ---------------- Feature Selection ----------------
# separating features and target variable
X = data.drop(labels=["fare_amount"], axis=1)
Y = data["fare_amount"].copy()


# splitting the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# feature extraction and scaling
PCA_featuers = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "jfk_dist",
    "ewr_dist",
    "lga_dist",
    "sol_dist",
    "nyc_dist",
]
pca_pipeline = Pipeline(
    [("scaler", RobustScaler()), ("pca", PCA(random_state=42, n_components=0.95))]
)

# Define column transformer
pca_colum_transformer = ColumnTransformer(
    transformers=[
        ("pca_features", pca_pipeline, PCA_featuers),
        (
            "scale_rest",
            RobustScaler(),
            [col for col in x_train.columns if col not in PCA_featuers],
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

# Transform
x_train_scaled = pca_colum_transformer.fit_transform(x_train)
x_test_scaled = pca_colum_transformer.transform(x_test)
# ---------------- Models ----------------
decision_tree_model = DecisionTreeRegressor(
    max_depth=15, min_samples_split=4, min_samples_leaf=2, random_state=42
)
decision_tree_model.fit(x_train_scaled, y_train)

ridge_model = Ridge(alpha=1.0, solver="auto", random_state=42)
ridge_model.fit(x_train_scaled, y_train)

# ---------------- Save Models ----------------
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

pickle.dump(
    decision_tree_model, open(os.path.join(models_dir, "decisiontree.pkl"), "wb")
)
pickle.dump(ridge_model, open(os.path.join(models_dir, "ridgeregression.pkl"), "wb"))

print("Models and transformer saved successfully!")
