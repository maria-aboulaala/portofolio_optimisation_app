import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np


st.title('Interface utilisateur:')


st.subheader('Entrez les parametres de votre portefeuille: :key: ')


with st.form(key="my_form"):



    tickers = st.multiselect('Coins', ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD','LTC-USD','USDT-USD','LINK-USD','ADA-USD','DOT-USD','BSV-USD','EOS-USD','ATOM-USD','SOL-USD','CRO-USD','XMR-USD'], default='BTC-USD')
    target_return = st.slider('Target Return', 0.0, 0.2, 0.1)
    st.form_submit_button("Simuler")



st.write('Vous avez choisi un portfeuille contenant:')
df = pd.DataFrame(tickers)
st.dataframe(df)

st.write('Informations concernant les valeurs du Close de chaque coins:')
data = yf.download(tickers)
st.write(data['Close'])
data1 = data['Close']
from scipy.optimize import minimize
# Define portfolio optimization function
def optimize_portfolio(data, target_return):
    n = len(data.columns)
    init_guess = [1/n]*n
    bounds = [(0,1) for i in range(n)]
    def portfolio_return(weights):
        returns = np.dot(data,weights)
        return -1*np.mean(returns)
    def portfolio_volatility(weights):
        returns = np.dot(data,weights)
        std = np.std(returns)
        return std
    def target_function(weights):
        return portfolio_volatility(weights)
    constraints = [{'type':'eq','fun':lambda x: np.sum(x)-1},
                   {'type':'eq','fun':lambda x: target_return-portfolio_return(x)}]
    results = minimize(target_function,init_guess,bounds=bounds,constraints=constraints)
    return results.x




# Optimize portfolio
optimal_weights = optimize_portfolio(data1, target_return)

# Show results
st.write('Optimal weights:', optimal_weights)