import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go



def getvol(ticker):
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

    # --- Graphique volatilité horaire ---
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(x=data.index, y=data["vol_60"], mode="lines", name="Vol 60h", line=dict(color="orange")))
    fig_hour.add_trace(go.Scatter(x=data.index, y=data["vol_24"], mode="lines", name="Vol 24h", line=dict(color="green", dash="dot")))
    fig_hour.update_layout(
        title=f"{ticker} - Volatilité horaire (1 an)",
        xaxis_title="Date",
        yaxis_title="Volatilité (écart-type)",
        template="plotly_dark",
        legend_title="Fenêtre"
    )
    fig_hour.show()

    # --- Graphique volatilité 15 minutes ---
    fig_minute = go.Figure()
    fig_minute.add_trace(go.Scatter(x=data_minute.index, y=data_minute["vol_60_minute"], mode="lines", name="Vol 60m", line=dict(color="cyan")))
    fig_minute.add_trace(go.Scatter(x=data_minute.index, y=data_minute["vol_24_minute"], mode="lines", name="Vol 24m", line=dict(color="magenta", dash="dot")))
    fig_minute.update_layout(
        title=f"{ticker} - Volatilité 15 min (1 mois)",
        xaxis_title="Date",
        yaxis_title="Volatilité (écart-type)",
        template="plotly_dark",
        legend_title="Fenêtre"
    )
    fig_minute.show()

tickers = ["","",""]

for ticker in tickers():
    getvol(ticker)

