import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import time

# Configuration
st.set_page_config(
    page_title="Sales Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv('../data/generated/campaign_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('../models/model.pkl')

# Sidebar
with st.sidebar:
    st.title("Paramètres")
    selected_hours = st.slider(
        "Plage horaire cible",
        min_value=9,
        max_value=18,
        value=(14, 16)
    )
    script_version = st.selectbox(
        "Version du script",
        options=['A', 'B', 'C']
    )
    min_duration = st.number_input(
        "Durée minimale (s)",
        min_value=60,
        value=180
    )

# Chargement
df = load_data()
model = load_model()

# Filtres
filtered_df = df[
    (df['call_time'].between(*selected_hours)) &
    (df['script_version'] == script_version) &
    (df['duration'] >= min_duration)
]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric(
    "Taux Conversion", 
    f"{filtered_df['converted'].mean()*100:.1f}%"
)
col2.metric(
    "Appels Qualifiés", 
    len(filtered_df)
)
col3.metric(
    "Durée Moyenne", 
    f"{filtered_df['duration'].mean():.0f}s"
)

# Visualisations
tab1, tab2 = st.tabs(["Performance Temporelle", "Analyse Régionale"])

with tab1:
    fig = px.line(
        filtered_df.groupby(['call_date', 'call_time'])
        ['converted'].mean().reset_index(),
        x='call_time',
        y='converted',
        color='call_date',
        title="Conversion par Heure"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    region_data = filtered_df.groupby('region').agg({
        'converted': 'mean',
        'duration': 'mean'
    }).reset_index()
    
    fig = px.bar(
        region_data,
        x='region',
        y='converted',
        hover_data=['duration'],
        title="Performance par Région"
    )
    st.plotly_chart(fig, use_container_width=True)

# Prédictions
st.header("Simulateur d'Appel")
with st.form("prediction_form"):
    duration = st.slider("Durée", 60, 600, 180)
    region = st.selectbox("Région", df['region'].unique())
    
    if st.form_submit_button("Prédire"):
        input_data = pd.DataFrame([{
            'duration_min': duration/60,
            'peak_hour': int(selected_hours[0] <= 16 <= selected_hours[1]),
            'region': region,
            'product': 'SaaS',
            'script_version': script_version,
            'day_of_week': 'Monday'
        }])
        
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"Probabilité de conversion : {proba*100:.1f}%")