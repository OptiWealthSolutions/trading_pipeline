import yfinance as yf
import pandas  as pd
import numpy as np
import fpdf 
import seaborn as sns
from fredapi import Fred
import plotly.graph_objects as go  
import matplotlib.pyplot as plt 



fred = Fred(api_key="e16626c91fa2b1af27704a783939bf72")

#structure and temrinal's logic

# params

start_date = "2020-01-01"

main_pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X"]

pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X", "NZDUSD=X","AUDUSD=X", 'EURAUD=X', "EURNZD=X", "EURCHF=X" ]

indices = {
    "DXY (Dollar Index)": "DTWEXBGS",      # Broad USD Index
    "EUR Index": "DEXUSEU",                 # USD/EUR (inversé pour index EUR)
    "GBP Index": "DEXUSUK",                 # USD/GBP (inversé pour index GBP)
    "JPY Index": "DEXJPUS",                 # JPY/USD
    "CHF Index": "DEXSZUS",                 # CHF/USD
    "CAD Index": "DEXCAUS",                 # CAD/USD
}

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
        data = yf.download(ticker, period="1y", interval="1h")
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

#plotPrice(["EURUSD=X"])

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
    data = yf.download(tickers, period="10y", interval='1d')['Close']
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

    
    # 1. Définir le style et les conteneurs de résultats
    plt.style.use('dark_background')
    
    all_means = {}  # Pour stocker les données du graphique
    ranking_results = [] # Pour stocker les données du DataFrame de sortie
    
    # Ordre correct des mois pour le tri
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    # 2. Boucler sur chaque ticker pour extraire les données
    for ticker in tickers:
        try:
            # Téléchargement des données mensuelles sur 10 ans
            data = yf.download(ticker, period="10y", interval="1mo").dropna()
            
            if data.empty:
                print(f"Aucune donnée pour {ticker}, passage au suivant.")
                continue
            
            # Calcul des rendements
            prices = data['Close']
            returns = prices.pct_change().dropna()
            data['return'] = returns
            data["Month"] = data.index.month_name()

            # Calcul du rendement moyen par mois
            mean_10y = data.groupby("Month")["return"].mean()
            
            # Tri des mois dans l'ordre chronologique (important !)
            mean_10y = mean_10y.reindex(month_order)
            
            # Stocker les données pour le graphique
            all_means[ticker] = mean_10y

            # Trouver le meilleur et le pire mois pour ce ticker
            best_month = mean_10y.idxmax()
            best_return = mean_10y.max()
            worst_month = mean_10y.idxmin()
            worst_return = mean_10y.min()
            
            # Ajouter au résumé
            ranking_results.append({
                "Ticker": ticker,
                "Best Month": best_month,
                "Best Return (%)": best_return * 100,
                "Worst Month": worst_month,
                "Worst Return (%)": worst_return * 100
            })

        except Exception as e:
            print(f"Erreur lors du traitement de {ticker}: {e}")

    # 3. Créer le graphique (Plot)
    if all_means:  # S'assurer qu'on a des données à tracer
        plt.figure(figsize=(14, 7))
        for ticker, mean in all_means.items():
            plt.plot(mean.index, mean.values * 100, marker='o', label=ticker)

        plt.title("Saisonnalité moyenne à 10 ans (tous actifs)", color='white')
        plt.xlabel("Mois", color='white')
        plt.ylabel("Rendement moyen (%)", color='white')
        plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45) # Améliore la lisibilité des mois
        plt.tight_layout()
        plt.show() # Affiche le graphique
    else:
        print("Aucune donnée de saisonnalité n'a pu être calculée pour le graphique.")

    # 4. Créer le DataFrame de sortie
    output_df = pd.DataFrame(ranking_results)
    if not output_df.empty:
        output_df = output_df.set_index("Ticker")
    
    return output_df

analyze_seasonality_for_pairs(main_pairs)

# graphique en scatter pour les relations entre paires et commodités sur différentes (timeframes si possible)

def getVix():
    data = yf.download("^VIX", period="3mo", interval="1h")['Close']
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
    x=data.index, 
    y=data["Close"], 
    mode="lines", 
    name="Prix de cloture", 
    line=dict(color="green", dash="dot")
        ))
    return data

getVix()

#sentiment de marché (twitter ?)

#volatilité et interpretation des datas 

