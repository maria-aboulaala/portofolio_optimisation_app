import streamlit as st
import pandas as pd
import yfinance as yf


st.title('Interface utilisateur:')
st.subheader('Entrez les parametres: :key: ')


with st.form(key="my_form"):   
    coins = st.multiselect('coins', ['Btc', 'uth', 'potatoes'])
    stock_name = st.selectbox(
    'Le symbole du stock',
    ('AAPL', 'MSFT', 'META', 'GOOG', 'AMZN'))    
    st.form_submit_button("Simuler")