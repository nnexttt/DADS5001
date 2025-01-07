import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

COLUMN_LABELS = {
    'occupancy_rate': 'Occupancy Rate',
    'no_tourist_occupied': 'Tourists in Occupied Rooms',
    'no_tourist_all': 'Total Tourists',
    'no_tourist_thai': 'Thai Tourists',
    'no_tourist_foreign': 'Foreign Tourists',
    'net_profit_all': 'Net Profit (Million baht)',
    'net_profit_thai': 'Thai Net Profit (Million baht)',
    'net_profit_foreign': 'Foreign Net Profit (Million baht)'
}

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def analyze_with_gemini(prompt):
    try:
        response = genai.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            max_output_tokens=300
        )
        return response.result
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def main():
    st.title("Descriptive Analysis and AI Insights for Tourism Data")

    
    file_path = "thailand_domestic_tourism.csv"  
    data = load_data(file_path)
    if data is None:
        return

    
    if "province_eng" not in data.columns:
        st.error("The data does not contain a 'province_eng' column for province selection.")
        return

    provinces = data["province_eng"].unique()
    selected_province = st.selectbox("Select a Province:", provinces)

    
    province_data = data[data["province_eng"] == selected_province]

    
    numeric_cols = [
        "occupancy_rate", "no_tourist_occupied", "no_tourist_all",
        "no_tourist_thai", "no_tourist_foreign", "net_profit_all",
        "net_profit_thai", "net_profit_foreign"
    ]
    for col in numeric_cols:
        if col in province_data.columns:
            province_data[col] = pd.to_numeric(province_data[col], errors="coerce")

    stats = province_data[numeric_cols].describe().T
    stats["median"] = province_data[numeric_cols].median()
    stats = stats.rename(index=COLUMN_LABELS)  
    st.subheader(f"Descriptive Statistics for {selected_province}")
    st.dataframe(stats)

    
    st.subheader("Key Highlights")
    st.write("การอธิบายจุดเด่นของข้อมูล:")
    key_highlights_prompt = (
        f"Analyze the descriptive statistics for the province '{selected_province}':\n"
        f"{stats[['mean', '50%', 'std', 'min', 'max']].to_string()}\n"
        "Identify key trends, anomalies, and interesting observations in the data."
    )
    try:
        key_highlights_response = genai.GenerativeModel("gemini-1.5-flash").generate_content([key_highlights_prompt])
        st.write(key_highlights_response.text)
    except Exception as e:
        st.error(f"Error generating key highlights: {str(e)}")

    
    st.subheader("Boxplot Analysis")
    st.write("แสดงการกระจายตัวของข้อมูลในแต่ละคอลัมน์:")
    try:
        
        renamed_data = province_data[numeric_cols].rename(columns=COLUMN_LABELS)
        
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=renamed_data, ax=ax)
        ax.set_title(f"Boxplot for Numeric Columns in {selected_province}")
        ax.set_ylabel("Values")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating boxplot: {str(e)}")

    
    recommendations_prompt = (
        f"Based on the descriptive statistics data for the province '{selected_province}':\n"
        f"{stats[['mean', '50%', 'std', 'min', 'max']].to_string()}\n"
        "Provide strategic recommendations to improve tourism performance in this province. "
        "Highlight potential growth opportunities and areas for improvement."
    )
    try:
        recommendations_response = genai.GenerativeModel("gemini-1.5-flash").generate_content([recommendations_prompt])
        st.write(recommendations_response.text)
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()
