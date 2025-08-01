import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open('xgb_model.json', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Booking Status Prediction")

# Sample inputs — change based on your dataset
lead_time  =  st.number_input("lead_time", min_value=0.0 , step=1.0)
avg_price = st.number_input("avg_price", min_value=0.0 , step=1.0)
# booking_status =  st.number_input("Booking Status", min_value=0.0 , step=1.0 , max_value=1.0)
total_nights =  st.number_input("Total Nights", min_value=0.0 , step=1.0)
total_member =  st.number_input("Total Member", min_value=0.0 , step=1.0)
total_repeat =  st.number_input("Total Repeat", min_value=0.0 , step=1.0 )
price_per_adult = st.number_input("Price Per Adult", step=1.0 , min_value=0.25)
has_special_requests = st.number_input("Has Special Requests", min_value=0.0 , step=1.0 , max_value=1.0)
high_price = st.number_input("High Price", min_value=0.0 , step=1.0 )
type_of_meal_Meal_Plan_2  = st.selectbox("Type Of meal_Meal Plan 2 :" , ["True","False"] )
room_type_Room_Type_4   = st.selectbox("room_type Room Type_4 :" , ["True","False"] )
room_type_Room_Type_5    = st.selectbox("room_type Room Type 5 :" , ["True","False"] )
room_type_Room_Type_6  = st.selectbox("room_type Room Type 6 :" , ["True","False"] )
room_type_Room_Type_7  = st.selectbox("room_type Room Type 7 :" , ["True","False"] )
market_segment_type_Complementary = st.selectbox("market Segment Type Complementary :" , ["True","False"] )
market_segment_type_Corporate =  st.selectbox("Market Segment Type Corporate :" , ["True","False"] )
market_segment_type_Offline = st.selectbox("Market Segment Type Offline :" , ["True","False"] )
market_segment_type_Online  = st.selectbox("Market Segment Type Online :" , ["True","False"] )


if st.button("Predict"):
    # Construct DataFrame (must match training format!)
    input_data = pd.DataFrame([{
        # "average price": avg_price,
         "lead time" :  lead_time ,
         "average price" : avg_price ,
        #  "Booking Status" : booking_status ,
         "total nights":total_nights ,
        "total_member" :total_member ,
         "total_repeat" :total_repeat ,
         "price_per_adult" : price_per_adult ,
         "has_special_requests" : has_special_requests ,
         "high_price":high_price ,
         "type of meal_Meal Plan 2":type_of_meal_Meal_Plan_2 ,
         "room type_Room_Type 4":room_type_Room_Type_4 ,
         "room type_Room_Type 5":room_type_Room_Type_5 ,
         "room type_Room_Type 6":room_type_Room_Type_6 ,
         "room type_Room_Type 7":room_type_Room_Type_7 ,
         "market segment type_Complementary":market_segment_type_Complementary ,
         "market segment type_Corporate":market_segment_type_Corporate ,
         "market segment type_Offline" : market_segment_type_Offline ,
         "market segment type_Online" :market_segment_type_Online

    }])

    # Scale and predict
    #  Convert boolean columns to integers
bool_cols = input_data.select_dtypes(include=['bool']).columns
input_data[bool_cols] = input_data[bool_cols].astype(int)

# ✅ Convert 'True'/'False' strings to 1/0
object_cols = input_data.select_dtypes(include='object').columns
for col in object_cols:
    if set(input_data[col].unique()) <= {'True', 'False'}:
        input_data[col] = input_data[col].map({'True': 1, 'False': 0})

# Then scale
input_scaled = scaler.transform(input_data)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

result = "Canceled" if prediction[0] == 1 else "Not Canceled"
st.success(f"Prediction: {result}")
