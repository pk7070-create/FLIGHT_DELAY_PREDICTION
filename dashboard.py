import streamlit as st

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Prediction")

st.write(
    "This application predicts whether a flight will be delayed "
    "using a Machine Learning model."
)

st.markdown("---")

st.header("📝 Flight Details (UI test)")


st.header("Enter flight Details")

col1,col2=st.columns(2)

with col1:
    month=st.selectbox(
        "Month",
        [1,2,3,4,5,6,7,8,9,10,11,12]
    )

    day_of_week=st.selectbox(
        "Day_of_week",
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    )

    distance=st.number_input(
        "Distance (miles)",
        min_value=0
    )

with col2:
    schedule_dep=st.number_input(
        "Schedule Departure Time (HHMM)",
        min_value=0,
        max_value=2359
    )

    airline=st.selectbox(
        "Airline",
        ["AS","B6","DL","EV","F9","OTHER"]
    )

    destination=st.selectbox(
        "Destination Airport",
        ["JFK","SFO","ORD","PHX","OTHER"]
    )


import joblib
import pandas as pd


feature_columns = joblib.load("feature_columns.pkl")
input_data = {col: 0 for col in feature_columns}

day_map={
    "Mon":1,
    "Tue":2,
    "Wed":3,
    "Thu":4,
    "Fri":5,
    "Sat":6,
    "Sun":7
}

day_of_week_num=day_map[day_of_week]
input_data["MONTH"]=month
input_data["DAY_OF_WEEK"] = day_of_week_num
input_data["DISTANCE"] = distance
input_data["SCHEDULED_DEPARTURE"] = schedule_dep


airline_col = f"AIRLINE_{airline}"

if airline_col in input_data:
    input_data[airline_col] = 1


dest_col = f"DESTINATION_AIRPORT_{destination}"

if dest_col in input_data:
    input_data[dest_col] = 1



input_df = pd.DataFrame([input_data])
st.write("Input shape:", input_df.shape)

model = joblib.load("flight_delay_model.pkl")



if st.button("Predict Delay"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⏰ Flight is likely to be DELAYED")
    else:
        st.success("✅ Flight is likely to be ON TIME")










