import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


def load_data(file_path):
    return pd.read_csv(file_path)

file_path = "thailand_domestic_tourism.csv"
data = load_data(file_path)

column_labels = {
    'occupancy_rate': 'Occupancy Rate',
    'no_tourist_occupied': 'Tourists in Occupied Rooms',
    'no_tourist_all': 'Total Tourists',
    'no_tourist_thai': 'Thai Tourists',
    'no_tourist_foreign': 'Foreign Tourists',
    'net_profit_all': 'Net Profit (Million baht)',
    'net_profit_thai': 'Thai Net Profit (Million baht)',
    'net_profit_foreign': 'Foreign Net Profit (Million baht)'}
data.rename(columns=column_labels, inplace=True)


st.title("Thailand net profit forecast")


st.write("Example data from CSV file:", data.head())


province = st.selectbox("Please select province:", data['province_eng'].unique())


prediction_type = st.selectbox(
    "Please select type of profit:",
    ["Thai Net Profit (Million baht)", "Foreign Net Profit (Million baht)"]
)


province_data = data[data['province_eng'] == province].reset_index(drop=True)


if st.button("Forecast"):
    
    province_data['month_year'] = pd.to_datetime(province_data['year'].astype(str) + '-' + province_data['month'].astype(str), format='%Y-%m')
    
    data_for_province = province_data[['month_year', prediction_type]]
    data_for_province = data_for_province.rename(columns={'month_year': 'ds', prediction_type: 'y'})

    
    model = Prophet()

    
    model.fit(data_for_province)

    
    future = model.make_future_dataframe(periods=12, freq='M')

    
    forecast = model.predict(future)

    
    st.write(f"Result of forecast in {province} ({prediction_type}):")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

    
    sns.set_theme(style="whitegrid")

    
    fig, ax = plt.subplots(figsize=(12, 6))
    model.plot(forecast, ax=ax)
    ax.set_title(f"Prediction of {prediction_type} in {province} (12 forecast)", fontsize=16, fontweight='bold')

    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"{prediction_type} (Million Baht)", fontsize=12)

    
    ax.grid(True, linestyle='--', alpha=0.7)

    
    for line in ax.get_lines():
        line.set_linewidth(2)

    
    ax.scatter(forecast['ds'].iloc[-12:], forecast['yhat'].iloc[-12:], color='red', label='Forecast (Last 12)', zorder=5)

    
    ax.legend(fontsize=12)

    
    st.pyplot(fig)

    
    st.write("Explanation result:")
    explanation_prompt = f"Explain the following forecast results for {province} and {prediction_type}:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_string(index=False)}"
    
    
    explanation_response = genai.GenerativeModel("gemini-1.5-flash").generate_content([explanation_prompt])
    
    
    st.write(explanation_response.text)