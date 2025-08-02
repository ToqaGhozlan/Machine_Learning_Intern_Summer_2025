# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 06:55:39 2025

@author: tuqam
"""

# === 1. استيراد المكتبات ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# === 2. تحميل البيانات ===
df = pd.read_csv("cleaned_data1.csv")

# === 3. حذف الأعمدة غير المفيدة ===
df.drop(columns=['User ID', 'User Name', 'Driver Name', 'key', 'pickup_datetime'], inplace=True)

# === 4. تقليل حجم الأعمدة لتقليل الذاكرة ===
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype('int8')

float_cols = df.select_dtypes(include='float64').columns
df[float_cols] = df[float_cols].astype('float32')

# === 5. تجهيز الميزات والهدف ===
X = df.drop(columns=['fare_amount'])
y = df['fare_amount'].astype('float32')

# === 6. تقسيم البيانات ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 7. تجربة موديل بسيط أولًا ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("🔹 Linear Regression:")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

# === 8. تجربة Random Forest ===
rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n🔹 Random Forest:")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))

# === 9. تجربة XGBoost ===
xgbr = xgb.XGBRegressor(n_estimators=20, max_depth=6, learning_rate=0.1, n_jobs=-1)
xgbr.fit(X_train, y_train)
y_pred_xgb = xgbr.predict(X_test)

print("\n🔹 XGBoost:")
print("R2 Score:", r2_score(y_test, y_pred_xgb))
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))

# === 10. Hyperparameter Tuning لموديل XGBoost (اختياري) ===
# param_grid = {
#     'max_depth': [3, 6],
#     'learning_rate': [0.01, 0.1],
#     'n_estimators': [100, 200]
# }
# grid = GridSearchCV(xgb.XGBRegressor(n_jobs=-1), param_grid, cv=3, scoring='r2', verbose=1)
# grid.fit(X_train, y_train)
# print("Best params:", grid.best_params_)
# print("Best R2:", grid.best_score_)
