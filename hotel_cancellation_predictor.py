# Import required libraries
import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
model = joblib.load("model_hotel_cancellation_prediction.pkl")
scaler = joblib.load("scaler_hotel_cancellation_prediction.pkl")
columns = joblib.load("columns_hotel_cancellation_prediction.pkl")
label_encoders = joblib.load("label_encoders_hotel_cancellation.pkl")

# Page configuration
st.set_page_config(page_title="Hotel Cancellation Prediction", page_icon="🏨", layout="centered")

# Title and description
st.title("🏨 Hotel Cancellation Prediction")
st.markdown("Please provide the customer's hotel booking information below to predict whether the booking is likely to be cancelled.")

# User Inputs
st.subheader("Booking Information")
col1, col2 = st.columns(2)

with col1:
    booking_date = st.date_input("Booking Date")
    arrival_date = st.date_input("Arrival Date")
    hotel = st.selectbox("Type of Hotel", ["City Hotel", "Resort Hotel"])
    adults = st.slider("Number of Adults", min_value=0, max_value=10, value=2)
    babies = st.slider("Number of Babies", min_value=0, max_value=10, value=0)
    weekend_nights = st.number_input("Nights in Weekend", min_value=0, max_value=50, value=0, step=1)
    week_nights = st.number_input("Nights in Weekdays", min_value=0, max_value=50, value=1, step=1)
    room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
    meal = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC"])

with col2:
    market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Groups", "Direct", 
                                                     "Corporate", "Complementary", "Aviation"])
    distribution_channel = st.selectbox("Distribution Channel", ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"])
    first_visit = st.selectbox("Is this the customer's first visit?", ["Yes", "No"])
    car_parking_spaces = st.slider("Required Car Parking Spaces", min_value=0, max_value=5, value=0)
    customer_type = st.selectbox("Customer Type", ["Contract", "Group", "Transient", "Transient-Party"])
    previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=100, 
                                                     value=0, step=1)
    days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=400, value=0, step=1)
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=30, value=0, step=1)
    booking_changes = st.number_input("Number of Booking Changes", min_value=0, max_value=20, value=0, step=1)
    special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=10, value=0, step=1)
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])

# High-cardinality categorical variables
st.subheader("Customer / Booking Source")

# Country
country_options = ["PRT", "GBR", "USA", "ESP", "IRL", "FRA", "ROU", "NOR", "OMN", "ARG", "POL", "DEU", "BEL", "CHE", "CN", "GRC", "ITA", "NLD", "DNK", "RUS", "SWE", "AUS", "EST", "CZE", "BRA", "FIN", "MOZ", "BWA", "LUX", "SVN", "ALB", "IND", "CHN", "MEX", "MAR", "UKR", "SMR", "LVA", "PRI", "SRB", "CHL", "AUT", "BLR", "LTU", "TUR", "ZAF", "AGO", "ISR", "CYM", "ZMB", "CPV", "ZWE", "DZA", "KOR", "CRI", "HUN", "ARE", "TUN", "JAM", "HRV", "HKG", "IRN", "GEO", "AND", "GIB", "URY", "JEY", "CAF", "CYP", "COL", "GGY", "KWT", "NGA", "MDV", "VEN", "SVK", "FJI", "KAZ", "PAK", "IDN", "LBN", "PHL", "SEN", "SYC", "AZE", "BHR", "NZL", "THA", "DOM", "MKD", "MYS", "ARM", "JPN", "LKA", "CUB", "CMR", "BIH", "MUS", "COM", "SUR", "UGA", "BGR", "CIV", "JOR", "SYR", "SGP", "BDI", "SAU", "VNM", "PLW", "QAT", "EGY", "PER", "MLT", "MWI", "ECU", "MDG", "ISL", "UZB", "NPL", "BHS", "MAC", "TGO", "TWN", "DJI", "STP", "KNA", "ETH", "IRQ", "HND", "RWA", "KHM", "MCO", "BGD", "IMN", "TJK", "NIC", "BEN", "VGB", "TZA", "GAB", "GHA", "TMP", "GLP", "KEN", "LIE", "GNB", "MNE", "UMI", "MYT", "FRO", "MMR", "PAN", "BFA", "LBY", "MLI", "NAM", "BOL", "PRY", "BRB", "ABW", "AIA", "SLV", "DMA", "PYF", "GUY", "LCA", "ATA", "GTM", "ASM", "MRT", "NCL", "KIR", "SDN", "ATF", "SLE", "LAO"]
country = st.selectbox("Country", country_options)

# Agent
agent_options = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0", "25.0", "26.0", "27.0", "28.0", "29.0", "30.0", "31.0", "32.0", "33.0", "34.0", "35.0", "36.0", "37.0", "38.0", "39.0", "40.0", "41.0", "42.0", "44.0", "45.0", "47.0", "50.0", "52.0", "53.0", "54.0", "55.0", "56.0", "57.0", "58.0", "59.0", "60.0", "61.0", "63.0", "64.0", "66.0", "67.0", "68.0", "69.0", "70.0", "71.0", "72.0", "73.0", "74.0", "75.0", "77.0", "78.0", "79.0", "81.0", "82.0", "83.0", "85.0", "86.0", "87.0", "88.0", "89.0", "90.0", "91.0", "92.0", "93.0", "94.0", "95.0", "96.0", "98.0", "99.0", "103.0", "104.0", "105.0", "106.0", "107.0", "110.0", "111.0", "112.0", "114.0", "115.0", "117.0", "118.0", "119.0", "121.0", "122.0", "126.0", "127.0", "128.0", "129.0", "132.0", "133.0", "134.0", "135.0", "138.0", "139.0", "141.0", "142.0", "143.0", "144.0", "146.0", "147.0", "148.0", "149.0", "150.0", "151.0", "152.0", "153.0", "154.0", "155.0", "156.0", "157.0", "158.0", "159.0", "162.0", "163.0", "165.0", "167.0", "168.0", "170.0", "171.0", "173.0", "174.0", "175.0", "177.0", "179.0", "180.0", "181.0", "182.0", "183.0", "184.0", "185.0", "187.0", "191.0", "192.0", "193.0", "195.0", "196.0", "197.0", "201.0", "205.0", "208.0", "210.0", "211.0", "213.0", "214.0", "215.0", "216.0", "219.0", "220.0", "223.0", "227.0", "229.0", "232.0", "234.0", "235.0", "236.0", "240.0", "241.0", "242.0", "243.0", "244.0", "245.0", "247.0", "248.0", "249.0", "250.0", "251.0", "252.0", "253.0", "254.0", "256.0", "257.0", "258.0", "261.0", "262.0", "265.0", "267.0", "269.0", "270.0", "273.0", "275.0", "276.0", "278.0", "280.0", "281.0", "282.0", "283.0", "285.0", "286.0", "287.0", "288.0", "289.0", "290.0", "291.0", "294.0", "295.0", "296.0", "298.0", "299.0", "300.0", "301.0", "302.0", "303.0", "304.0", "305.0", "306.0", "307.0", "308.0", "310.0", "313.0", "314.0", "315.0", "321.0", "323.0", "324.0", "325.0", "326.0", "327.0", "328.0", "330.0", "331.0", "332.0", "333.0", "334.0", "335.0", "336.0", "337.0", "339.0", "341.0", "344.0", "346.0", "348.0", "350.0", "352.0", "354.0", "355.0", "358.0", "359.0", "360.0", "363.0", "364.0", "367.0", "368.0", "370.0", "371.0", "375.0", "378.0", "384.0", "385.0", "387.0", "388.0", "390.0", "391.0", "393.0", "394.0", "397.0", "403.0", "404.0", "405.0", "406.0", "408.0", "410.0", "411.0", "414.0", "416.0", "418.0", "420.0", "423.0", "425.0", "426.0", "427.0", "429.0", "430.0", "431.0", "432.0", "433.0", "434.0", "436.0", "438.0", "440.0", "441.0", "444.0", "446.0", "449.0", "450.0", "451.0", "453.0", "454.0", "455.0", "459.0", "461.0", "464.0", "467.0", "468.0", "469.0", "472.0", "474.0", "475.0", "476.0", "479.0", "480.0", "481.0", "483.0", "484.0", "492.0", "493.0", "495.0", "497.0", "502.0", "508.0", "509.0", "510.0", "526.0", "527.0", "531.0", "535.0"]
agent = st.selectbox("Select Agent ID", agent_options)

# Validate dates
if arrival_date < booking_date:
    st.error("Arrival date cannot be before booking date.")
    st.stop()

# Calculate derived features
lead_time = (arrival_date - booking_date).days
arrival_year = arrival_date.year
arrival_month = arrival_date.strftime("%B")
arrival_week_number = arrival_date.isocalendar().week
is_repeated_guest = 1 if first_visit == "No" else 0

# Convert agent to string because LabelEncoder was trained using the original categorical values.
agent = str(agent)

# Label Encoding
try:
    country_encoded = label_encoders["country"].transform([country])[0]
    month_encoded = label_encoders["arrival_date_month"].transform([arrival_month])[0]
    agent_value = float(agent)
    agent_encoded = label_encoders["agent"].transform([agent_value])[0]

except ValueError as e:
    st.error(f"One of the selected values was not present during model training: {e}")
    st.stop()

# Create raw input dictionary
raw_input = {"adults": adults,
            "hotel_" + hotel: 1,
            "lead_time": lead_time,
            "days_in_waiting_list": days_in_waiting_list,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "country": country_encoded,
            "agent": agent_encoded,
            "previous_cancellations": previous_cancellations,
            "babies": babies,
            "booking_changes": booking_changes,
            "required_car_parking_spaces": car_parking_spaces,
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights,
            "total_of_special_requests": special_requests,
            "is_repeated_guest": is_repeated_guest,
            "arrival_date_month": month_encoded,
            "arrival_date_week_number": arrival_week_number,
            "meal_" + meal: 1,
            "arrival_date_year_" + str(arrival_year): 1,
            "market_segment_" + market_segment: 1,
            "distribution_channel_" + distribution_channel: 1,
            "reserved_room_type_" + room_type: 1,
            "deposit_type_" + deposit_type: 1,
            "customer_type_" + customer_type: 1}

# Create DataFrame
input_df = pd.DataFrame([raw_input])

# Set the other features to 0
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# Feature Scaling
numerical_cols = ["lead_time", "previous_bookings_not_canceled", "days_in_waiting_list", "adults", 
                  "previous_cancellations", "babies", "booking_changes", 
                  "required_car_parking_spaces", "stays_in_weekend_nights", 
                  "stays_in_week_nights", "total_of_special_requests"]
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Prediction
st.markdown("---")
if st.button("🔮 Predict Cancellation"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    probability_percentage = probability * 100
    if prediction == 1:
        st.error(f"⚠️ This booking is predicted to be CANCELLED.")
    else:
        st.success(f"✅ This booking is predicted to NOT be cancelled.")
    st.metric("Cancellation Probability", f"{probability_percentage:.2f}%")

# Disclaimer
st.markdown("---")
st.caption("Disclaimer: This application is intended for educational purposes only. Predictions are generated by a machine learning model and should not be used as a substitute for professional advice.")