import streamlit as st
import pandas as pd
import yfinance as yf

st.title('Optimisation d un portfeuille de crypto utilisant le machine learning')



st.subheader('Entrez les parametres: :key: ')


with st.form(key="my_form"):   
    stock_name = st.selectbox(
    'Le symbole du stock',
    ('AAPL', 'MSFT', 'META', 'GOOG', 'AMZN'))    
    st.form_submit_button("Simuler")















st.markdown(
    """
---
 Realisé par 
 > Aboulaala Maria |  Aberhouch Anass | Mahdioui Med Amine

    """
)

