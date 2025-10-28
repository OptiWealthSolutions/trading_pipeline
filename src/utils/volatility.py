import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = "AAPL"

data = yf.download(ticker,period="1y",interval="1h")
data['vol'] = data['Close'].std()
data['vol_60'] = data['Close'].rolling(window=60).std()
data['vol_24'] = data['Close'].rolling(window=24).std()


data_minute = yf.download(ticker,period="1mo",interval="15m")
data_minute['vol'] = data['Close'].std()
data_minute['vol_60_minute'] = data_minute['Close'].rolling(window=4).std()
data_minute['vol_24_minute'] = data_minute['Close'].rolling(window=2).std()

#pct du high
high_60 = data['vol_60'].max()
print(high_60)
pct_max_60 = data['vol_60'].iloc[-1]/high_60
print(round((pct_max_60*100),2),"%")


high_60_minute = data_minute['vol_60_minute'].max()
print(high_60)
pct_max_60_minute = data_minute['vol_60_minute'].iloc[-1]/high_60
print(round((pct_max_60_minute*100),2),"%")

plt.style.use('dark_background')
plt.plot(data['vol_60'])
plt.plot(data["vol_24"],alpha=0.5,color='green')
plt.show()

plt.plot(data_minute['vol_60_minute'])
plt.plot(data_minute["vol_24_minute"],alpha=0.5,color='green')
plt.show()

plt.show()
