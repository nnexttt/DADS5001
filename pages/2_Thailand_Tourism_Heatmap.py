import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
data = pd.read_csv("thailand_domestic_tourism.csv")

name_mapping = {
    "Bangkok": "Bangkok Metropolis",
    "Prachinburi": "Prachin Buri",
    "Buriram": "Buri Ram",
    "Sisaket": "Si Sa Ket",
    "Chainat": "Chai Nat",
    "Lopburi": "Lop Buri",
    "Nong Bua Lamphu": "Nong Bua Lam Phu"
}
data["province_eng"] = data["province_eng"].replace(name_mapping)

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

st.title("Thailand Heatmap")

column_options = list(column_labels.keys())
selected_column = st.selectbox("Select a column to display on the heatmap:", column_options, format_func=lambda x: column_labels[x])

years = data["year"].unique()
selected_year = st.selectbox("Select a year:", sorted(years))

include_bangkok = st.checkbox("Include Bangkok Metropolis in calculations", value=True)

filtered_data = data[data["year"] == selected_year]

if not include_bangkok:
    filtered_data = filtered_data[filtered_data["province_eng"] != "Bangkok Metropolis"]

if selected_column == "occupancy_rate":
    filtered_data[selected_column] = pd.to_numeric(data[selected_column], errors="coerce")
    aggregated_data = filtered_data.groupby("province_eng")[selected_column].mean().reset_index()
else:
    filtered_data[selected_column] = pd.to_numeric(data[selected_column], errors="coerce")
    aggregated_data = filtered_data.groupby("province_eng")[selected_column].sum().reset_index()
aggregated_data.rename(columns={selected_column: "value"}, inplace=True)

data_geo = aggregated_data 

fig = px.choropleth(
    data_geo,
    geojson="https://raw.githubusercontent.com/apisit/thailand.json/master/thailand.json",
    featureidkey="properties.name",
    locations="province_eng",
    color="value",
    color_continuous_scale="Viridis",
    title=f"Heatmap of {selected_column} in {selected_year} by Province"
)
fig.update_geos(fitbounds="locations", visible=False)

st.plotly_chart(fig)

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key="GOOGLE_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

st.subheader("Chart Explanation")
def get_google_explanation(data_geo, selected_column, selected_year):
    prompt = f"""
    Analyze the following heatmap data for Thailand provinces in the year {selected_year}:
    Column: {selected_column}
    Data: {data_geo.to_dict()}

    Provide insights about which provinces stand out and why, trends, or anomalies based on the data.
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit button for AI explanation
if st.button("Explain the Chart with Google AI"):
    try:
        explanation = get_google_explanation(aggregated_data, selected_column, selected_year)
        st.subheader("AI-Powered Insights (Google Generative AI)")
        st.write(explanation)
    except Exception as e:
        st.error(f"An error occurred: {e}")