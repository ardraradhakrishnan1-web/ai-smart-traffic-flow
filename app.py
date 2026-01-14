import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Smart Traffic Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: navy;'>AI-Based Smart Traffic Signal Optimization Dashboard</h1>", unsafe_allow_html=True)

# Load model
with open("traffic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset
df = pd.read_csv("traffic.csv")

# DateTime processing
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['Year'] = df['DateTime'].dt.year

# Sidebar inputs
st.sidebar.header("Traffic Prediction")

hour = st.sidebar.slider("Hour", 0, 23, 10)
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.selectbox("Year", sorted(df['Year'].unique()))
junction = st.sidebar.selectbox("Junction", sorted(df['Junction'].unique()))

if st.sidebar.button("Predict Traffic"):
    input_data = [[hour, day, month, year, junction]]
    prediction = model.predict(input_data)
    st.sidebar.success(f"Predicted Traffic: {int(prediction[0])}")

# KPIs
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Traffic", int(df['Vehicles'].sum()))
with col2:
    st.metric("Average Traffic", round(df['Vehicles'].mean(), 2))

# Charts
col3, col4 = st.columns(2)

with col3:
    st.subheader("Vehicles by Junction")
    junction_data = df.groupby('Junction')['Vehicles'].sum()
    fig, ax = plt.subplots()
    junction_data.plot(kind='line', marker='o', ax=ax)
    st.pyplot(fig)

with col4:
    st.subheader("Monthly Traffic Analysis")
    month_data = df.groupby('Month')['Vehicles'].sum()
    fig, ax = plt.subplots()
    month_data.plot(kind='bar', ax=ax)
    st.pyplot(fig)

col5, col6 = st.columns(2)

with col5:
    st.subheader("Daily Traffic Analysis")
    day_data = df.groupby('Day')['Vehicles'].sum()
    fig, ax = plt.subplots()
    day_data.plot(kind='bar', ax=ax)
    st.pyplot(fig)

with col6:
    st.subheader("Yearly Traffic Trend")
    year_data = df.groupby('Year')['Vehicles'].sum()
    fig, ax = plt.subplots()
    year_data.plot(kind='line', marker='o', ax=ax)
    st.pyplot(fig)
