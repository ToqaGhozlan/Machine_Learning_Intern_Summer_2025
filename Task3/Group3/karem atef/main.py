import streamlit as st
import pandas as pd
import requests

st.title("🧠 Hotel Booking Cancellation Prediction")
st.write("أدخل بيانات الحجز يدويًا وسيتم التنبؤ إذا كان الحجز سيتم إلغاؤه أم لا")

# ====== إدخال الأعمدة واحدة واحدة ======

type_of_meal = st.selectbox("نوع الوجبة (type of meal)", [0, 1, 2, 3])
repeated = st.selectbox("هل العميل مكرر؟ (repeated)", [0, 1])
lead_time = st.number_input("المدة قبل الحجز (lead time)", min_value=0, max_value=365)
market_segment = st.selectbox("نوع السوق (market segment type)", [0, 1, 2, 3, 4, 5, 6])
average_price = st.number_input("متوسط السعر (average price)", min_value=0.0, max_value=1000.0)
special_requests = st.number_input("عدد الطلبات الخاصة (special requests)", min_value=0, max_value=5)

# ====== تحويل القيم إلى DataFrame ======
data = {
    "type of meal": [type_of_meal],
    "repeated": [repeated],
    "lead time": [lead_time],
    "market segment type": [market_segment],
    "average price ": [average_price],
    "special requests": [special_requests]
}

df = pd.DataFrame(data)

# ====== إرسال البيانات إلى Flask ======
if st.button("🔍 تنبأ"):
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=df.to_dict(orient="records"))
        result = response.json()

        if "prediction" in result:
            prediction = result['prediction'][0]
            label = "Canceled ❌" if prediction == 1 else "Not Canceled ✅"
            st.success(f"النتيجة: {label}")
        else:
            st.error(f"خطأ في الرد: {result}")

    except Exception as e:
        st.error(f"لم يتم الاتصال بسيرفر Flask. تأكد أنه شغال.\nالخطأ: {e}")
