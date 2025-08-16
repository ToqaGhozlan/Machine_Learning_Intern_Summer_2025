import pandas as pd
import numpy as np
import joblib

# تحميل الموديل المحفوظ
_art = joblib.load("model.pkl")
_PIPE = _art["pipeline"]      # prep(OHE+Scale) -> SMOTE(في fit فقط) -> Poly -> LR
_FEATURES = _art["features"]  # أعمدة X الخام وقت التدريب

def _to_datetime(v):
    try:
        return pd.to_datetime(v, errors="coerce")
    except Exception:
        return pd.NaT

def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    نبني صفًا واحدًا بنفس أعمدة X التي درّبنا عليها.
    لو تستخدم أعمدة مشتقة (lead time, booking_year/month) أضفها هنا.
    """
    df = pd.DataFrame([user_input])

    # مثال: حساب lead time إن كانت التواريخ موجودة
    if {"date of reservation", "arrival_date"}.issubset(df.columns):
        d1 = _to_datetime(df.at[0, "date of reservation"])
        d2 = _to_datetime(df.at[0, "arrival_date"])
        if pd.notna(d1) and pd.notna(d2):
            df["lead time"] = max((d2 - d1).days, 0)
        df.drop(columns=["arrival_date"], inplace=True, errors="ignore")

    # مثال: year/month لو كانت ضمن ميزات التدريب
    if "date of reservation" in df.columns and (
        "booking_year" in _FEATURES or "booking_month" in _FEATURES
    ):
        d = _to_datetime(df.at[0, "date of reservation"])
        if pd.notna(d):
            df["booking_year"] = d.year
            df["booking_month"] = d.month
    if "date of reservation" in df.columns and "date of reservation" not in _FEATURES:
        df.drop(columns=["date of reservation"], inplace=True, errors="ignore")

    # مطابقة الأعمدة (نضيف الناقص بـ 0 ونرتب بحسب _FEATURES)
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[_FEATURES]

    return df

def predict(user_input: dict):
    """
    يُرجع:
      - label: 0/1 (booking status)
      - prob: احتمالية الانتماء للفئة 1 (إن وُجدت)
    """
    X = preprocess_input(user_input)
    label = int(_PIPE.predict(X)[0])

    prob = None
    if hasattr(_PIPE, "predict_proba"):
        try:
            prob = float(_PIPE.predict_proba(X)[0][1])  # احتمال الفئة 1
        except Exception:
            prob = None

    return {"label": label, "prob": prob}
