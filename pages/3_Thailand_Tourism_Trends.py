import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

data = pd.read_csv('thailand_domestic_tourism.csv')

st.title("Thailand Tourism Trends :chart_with_upwards_trend:")

st.markdown("""
    This interactive dashboard provides an in-depth analysis of trends across various provinces in Thailand. 
    Users can select a province and a specific metric **to explore monthly trends from 2019 to 2024.**
""")


st.sidebar.header("Filter Options")

provinces = data['province_eng'].unique()
selected_province = st.sidebar.selectbox("**Select a Province**", provinces)

column_labels = {
    'occupancy_rate': 'Occupancy Rate',
    'no_tourist_occupied': 'Tourists in Occupied Rooms',
    'no_tourist_all': 'Total Tourists',
    'no_tourist_thai': 'Thai Tourists',
    'no_tourist_foreign': 'Foreign Tourists',
    'net_profit_all': 'Net Profit (Million baht)',
    'net_profit_thai': 'Thai Net Profit (Million baht)',
    'net_profit_foreign': 'Foreign Net Profit (Million baht)'
}
columns_to_plot = st.sidebar.selectbox("**Select Factor**", options=list(column_labels.keys()), format_func=lambda x: column_labels[x])

filtered_data = data[data['province_eng'] == selected_province]
filtered_data['year_month'] = filtered_data['year'].astype(str) + '-' + filtered_data['month'].astype(str).str.zfill(2)

numeric_columns = filtered_data[[columns_to_plot]].apply(pd.to_numeric, errors='coerce')
filtered_data[columns_to_plot] = numeric_columns

yearly_data = filtered_data.copy()


st.header(f":green[Trend Graphs:] {selected_province}")
fig = px.line(
    yearly_data,
    x='year_month',
    y=columns_to_plot,
    title=f"Trends in {selected_province} (2019-2024)",
    labels={'year_month': 'Month-Year', columns_to_plot: column_labels[columns_to_plot]}
)
fig.update_xaxes(tickmode='array', tickvals=[f"{year}-01" for year in range(2019, 2025)], ticktext=[str(year) for year in range(2019, 2025)], title_text='Year')
fig.update_yaxes(title_text='Values')
st.plotly_chart(fig)

def generate_summary(data, selected_province, columns_to_plot, column_labels):
    """Generates a summary of the chart data using Generative AI."""
    prompt = f"""
    Analyze the tourism data for {selected_province} from 2019 to 2024. 
    The data includes the following metric: {column_labels[columns_to_plot]}. 
    Provide a concise summary of the key trends, patterns, or anomalies observed.
    
    Data (Year-Month, Value):
    """
    for index, row in data.iterrows():
        prompt += f"{row['year_month']}, {row[columns_to_plot]}\n"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "An error occurred while generating the summary."


with st.spinner("Generating summary..."):  # Show a spinner while generating
    summary = generate_summary(yearly_data, selected_province, columns_to_plot, column_labels)

st.subheader("Summary:")
st.write(summary)