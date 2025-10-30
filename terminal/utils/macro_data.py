import yfinance as yf
import pandas  as pd
import numpy as np
import fpdf 
import seaborn as sns
from fredapi import Fred
import plotly.graph_objects as go  
import matplotlib.pyplot as plt 
import sys 
from config import macro_data_config, api_key


fred = Fred(api_key=api_key)

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