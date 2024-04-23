#!/usr/bin/env python
# coding: utf-8


#------------------------------------loading data--------------------------------------------------#
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data
@st.cache
def load_data():
    data = pd.read_csv('https://github.com/tecnest/analysis-application/raw/main/ITEMS%26COLOR.xlsx')
    return data

data = load_data()

# Sidebar for user inputs
st.sidebar.header('User Input Features')
selected_store = st.sidebar.selectbox('Select Store', data['store'].unique())


filtered_data = data[data['store'] == selected_store]

# Display data or summary views
st.header('Data View')
st.write(filtered_data)

# Visualization
st.header('Sales by Product')
fig = px.bar(filtered_data, x='product', y='sales')
st.plotly_chart(fig)