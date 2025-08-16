import os
import pandas as pd
import numpy as np

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# model imports
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle

# dataset read
path = r"C:\Users\Malak\Internships Summer 2025\Cellula Technologies Internship\Week (4)\final_internship_data.csv"
data = pd.read_csv(path)


# data preprocessing
# removing invalid values
data = data.drop(data[data["fare_amount"] > 50][data["distance"] < 10].index, axis=0)
data = data.drop(data[data["distance"] > 50].index, axis=0)
data = data.drop(data[data["distance"] < 0.1].index, axis=0)
data = data.drop(data[data["fare_amount"] < 1].index, axis=0)


# removing unnecessary columns
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


# removing rows with null values
data.dropna(axis=0, inplace=True)

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
# To DataFrame
x_train_scaled = pd.DataFrame(
    x_train_scaled, columns=pca_colum_transformer.get_feature_names_out()
)
# Fit the model
# model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=20,
#     min_samples_split=5,
#     min_samples_leaf=4,
#     n_jobs=-1,
#     random_state=42,
# )
# model.fit(x_train_scaled, y_train)

# Fit the XGBRegressor model
model2 = XGBRegressor(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.8,
    reg_lambda=5,
    reg_alpha=1,
    gamma=1,
    colsample_bytree=0.8,
    random_state=42,
)
model2.fit(x_train_scaled, y_train)

# fit the linear regression model
model3 = LinearRegression()
model3.fit(x_train_scaled, y_train)


# # Fit the KNeighborsRegressor model
model4 = KNeighborsRegressor()
model4.fit(x_train_scaled, y_train)


# ---------------- Save Models ----------------

# pickle.dump(pca_colum_transformer, open("transformer.pkl", "wb"))
# #pickle.dump(model, open("RandomForest.pkl", "wb"))
# pickle.dump(model4, open("KNN.pkl", "wb"))
# pickle.dump(model2, open("XGB.pkl", "wb"))
# pickle.dump(model3, open("Linear.pkl", "wb"))


models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

#--------Changed--------
# # pickle.dump(pca_pipeline, open(os.path.join(models_dir, "transformer2.pkl"), "wb"))
# pickle.dump(pca_colum_transformer, open(os.path.join(models_dir, "transformer2.pkl"), "wb"))
#--------to-------
pickle.dump(pca_colum_transformer, open(os.path.join(models_dir, "transformer2.pkl"), "wb"))
#-----------------

pickle.dump(model2, open(os.path.join(models_dir, "XGB.pkl"), "wb"))
pickle.dump(model3, open(os.path.join(models_dir, "Linear.pkl"), "wb"))
pickle.dump(model4, open(os.path.join(models_dir, "KNN.pkl"), "wb"))

print("Model2 pipeline and models saved successfully!")

