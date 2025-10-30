import yfinance as yf
import pandas  as pd
import numpy as np
import fpdf 
import seaborn as sns
from fredapi import Fred
import plotly.graph_objects as go  
import matplotlib.pyplot as plt 
from config import main_pairs, pairs, config

def plotPrice(tickers):
    for ticker in tickers:
        data = yf.download(ticker, period="1y", interval="1h",progress=False)
        data["SMA_200"] = data['Close'].rolling(window=200).mean()
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Chandelier"  # Nom pour la légende
        ))
        fig_price.add_trace(go.Scatter(
            x=data.index, 
            y=data["SMA_200"], 
            mode="lines", 
            name="SMA 200", 
            line=dict(color="green", dash="dot")
        ))
        fig_price.update_layout(
            title=f"{ticker} - Prix {ticker} en H1", 
            xaxis_title="Date",
            yaxis_title="Prix",
            template="plotly_dark",
            legend_title="Légende", 
            xaxis_rangeslider_visible=False
        )
        
        fig_price.show()


# Matrice de correlation entre chaque paires et plot des indices de chaques monnaies et saisonnalité
def plotCurrenciesIndex(tickers):
    data = {}
    for name,ticker in tickers.items():
        serie = fred.get_series(ticker, observation_start=start_date)
        if serie is not None and len(serie) > 0:
            data[name] = serie
    df = pd.DataFrame(data)
    df = df.ffill().dropna(how='all')

    # Normaliser chaque série à 100 au début de la période
    df_normalized = (df / df.iloc[0]) * 100
    fig = go.Figure()
    # Couleurs pour chaque devise
    colors = {
        "DXY (Dollar Index)": "#00ff41",
        "EUR Index": "#ff006e",
        "GBP Index": "#ffbe0b",
        "JPY Index": "#8338ec",
        "CHF Index": "#fb5607",
        "CAD Index": "#3a86ff"
    }
    for col in df_normalized.columns:
        fig.add_trace(go.Scatter(x=df_normalized.index,y=df_normalized[col],name=col,line=dict(width=2.5, color=colors.get(col, "#ffffff")),mode='lines',hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Date: %{x}<br>' +
                        'Valeur: %{y:.2f}<br>' +
                        '<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': "Indices de Monnaie - Normalisés Base 100",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Date",
        yaxis_title="Indice (Base 100 au début)",
        template="plotly_dark",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        height=700,
        font=dict(size=12)
    )
    fig.add_hline(
        y=100, 
        line_dash="dash", 
        line_color="red", 
        opacity=0.5,
        annotation_text="Base 100",
        annotation_position="right"
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
    fig.show()
    return df.iloc[-1]

def corrMatrix(tickers):
    data = yf.download(tickers, period="10y", interval='1d',progress=False)['Close']
    returns = data.pct_change()
    returns = returns.dropna()
    corr_matrix = round(returns.corr(),2)
    plt.figure(figsize=(12, 9)) # Ajuster la taille pour la lisibilité
    sns.heatmap(
        corr_matrix, 
        annot=True,     # Afficher les valeurs dans les cases
        cmap='coolwarm',  # Palette de couleurs (bleu = faible, rouge = forte)
        fmt='.2f',      # Formater les nombres à 2 décimales
        linewidths=0.5    # Ajouter de fines lignes entre les cases
    )
    plt.title('Matrice de Corrélation des Rendements Journaliers (10 ans)')
    plt.show()
    most_correlated = corr_matrix.max()<1
    print(most_correlated)


def analyze_seasonality_for_pairs(tickers: list):
    plt.style.use('dark_background')
    
    all_means = {} 
    ranking_results = [] 
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="10y", interval="1mo",progress=False).dropna()
            
            if data.empty:
                print(f"Aucune donnée pour {ticker}, passage au suivant.")
                continue
            prices = data['Close']
            returns = prices.pct_change().dropna()
            data['return'] = returns
            data["Month"] = data.index.month_name()

            mean_10y = data.groupby("Month")["return"].mean()
            mean_10y = mean_10y.reindex(month_order)
            
            all_means[ticker] = mean_10y

            best_month = mean_10y.idxmax()
            best_return = mean_10y.max()
            worst_month = mean_10y.idxmin()
            worst_return = mean_10y.min()
            
            ranking_results.append({
                "Ticker": ticker,
                "Best Month": best_month,
                "Best Return (%)": best_return * 100,
                "Worst Month": worst_month,
                "Worst Return (%)": worst_return * 100
            })

        except Exception as e:
            print(f"Erreur lors du traitement de {ticker}: {e}")


    if all_means: 
        plt.figure(figsize=(14, 7))
        for ticker, mean in all_means.items():
            plt.plot(mean.index, mean.values * 100, marker='o', label=ticker)

        plt.title("Saisonnalité moyenne à 10 ans (tous actifs)", color='white')
        plt.xlabel("Mois", color='white')
        plt.ylabel("Rendement moyen (%)", color='white')
        plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45) 
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune donnée de saisonnalité n'a pu être calculée pour le graphique.")


    output_df = pd.DataFrame(ranking_results)
    if not output_df.empty:
        output_df = output_df.set_index("Ticker")
    
    return output_df

analyze_seasonality_for_pairs(main_pairs)

# graphique en scatter pour les relations entre paires et commodités sur différentes (timeframes si possible)

def getVix():
    data = yf.download("^VIX", period="3mo", interval="1h", progress=False)
    
    if data.empty:
        print("Impossible de récupérer les données du VIX.")
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode="lines",
        name="VIX (Volatilité implicite du S&P 500)",
        line=dict(color="#00ff41", width=2)
    ))

    fig.update_layout(
        title="Indice VIX - Évolution sur 3 mois",
        xaxis_title="Date",
        yaxis_title="Valeur du VIX",
        template="plotly_dark",
        hovermode='x unified',
        height=600
    )
    # On ne l'affiche pas ici, on le retourne
    return fig

getVix()