import yfinance as yf
import pandas  as pd
import numpy as np
import plotly as plt
import fpdf 
import seaborn as sns
from fredapi import Fred
import plotly.graph_objects as go

fred = Fred(api_key="e16626c91fa2b1af27704a783939bf72")

#structure and temrinal's logic

# params
main_pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X"]
pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X", "NZDUSD=X","AUDUSD=X", 'EURAUD=X', "EURNZD=X", "EURCHF=X" ]
# Macro data frame for the principal monetary area (japan, usa, euro, oceania, china, canada)

macro_data_config = {
    'USD': {
        'CPI': 'CPIAUCSL',
        'GDP': 'GDP',
        'PPI': 'PPIACO',
        'Interest Rate': 'FEDFUNDS',
        'Jobless Rate': 'UNRATE',
        'GDP Growth': 'A191RL1Q225SBEA'
    },
    'EUR': {
        'CPI': 'CP0000EZ19M086NEST',
        'GDP': 'CLVMNACSCAB1GQEA19',
        'PPI': 'PPI_EA19',
        'Interest Rate': 'ECBDFR',
        'Jobless Rate': 'LRHUTTTTEZM156S',
        'GDP Growth': 'NAEXKP01EZQ657S'
    },
    'GBP': {
        'CPI': 'CPALTT01GBM657N',
        'GDP': 'NAEXKP01GBQ661S',
        'PPI': 'PPIACOGBM086NEST',
        'Interest Rate': 'BOEBASE',
        'Jobless Rate': 'LRHUTTTTGBM156S',
        'GDP Growth': 'NAEXKP01GBQ657S'
    },
    'JPY': {
        'CPI': 'CPALTT01JPM657N',
        'GDP': 'CLVMNACSCAB1GQJP',
        'PPI': 'PPIACOJPM086NEST',
        'Interest Rate': 'IRSTCB01JPM156N',
        'Jobless Rate': 'LRHUTTTTJPQ156S',
        'GDP Growth': 'NAEXKP01JPQ657S'
    },
    'CAD': {
        'CPI': 'CPALTT01CAM657N',
        'GDP': 'CLVMNACSCAB1GQCA',
        'PPI': 'PPIACOCAM086NEST',
        'Interest Rate': 'IRSTCB01CAM156N',
        'Jobless Rate': 'LRHUTTTTCAQ156S',
        'GDP Growth': 'NAEXKP01CAQ657S'
    }
}

def fetchMacroData():
    results = []

    for devise, codes in macro_data_config.items():
        row = {'Devise': devise}
        for indicator, code in codes.items():
            try:
                serie = fred.get_series(code)
                if not serie.empty:
                    row[indicator] = serie.iloc[-1]  # dernière donnée disponible
                else:
                    row[indicator] = np.nan
            except Exception:
                row[indicator] = np.nan
        results.append(row)

    macro_df = pd.DataFrame(results).set_index('Devise')
    return macro_df

#fetchMacroData()

# Situation globale sur les paires principales (plot avec 4/6 graph max), leur tendance sur plusieurs time frames
def plotPrice(tickers):
    for ticker in tickers:
        data = yf.download(ticker,period="1y",interval="1h")
        data["SMA_200"] = data['Close'].rolling(window=200).mean()
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Prix ", line=dict(color="orange")))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(color="green", dash="dot")))
        fig_price.update_layout(
            title=f"{ticker} - Prix {ticker} en H4",
            xaxis_title="Date",
            yaxis_title="Prix",
            template="plotly_dark",
            legend_title="Fenêtre"
        )
        fig_price.show()

plotPrice(["EURUSD=X"])
# Matrice de correlation entre chaque paires et plot des indices de chaques monnaies et saisonnalité

# graphique en scatter pour les relations entre paires et commodités sur différentes (timeframes si possible)

#sentiment de marché (twitter ?)

#volatilité et interpretation des datas 

