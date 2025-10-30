import os
import re
from datetime import datetime, timedelta
import shiny
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Clé API FRED (de terminal.py) ---
# Il est préférable de la charger depuis les variables d'environnement aussi
# fred_api_key = os.environ.get("FRED_API_KEY", "e16626c91fa2b1af27704a783939bf72")
# Par simplicité, nous gardons celle du fichier :
fred = Fred(api_key="e16626c91fa2b1af27704a783939bf72")

# --- Listes de paires (de terminal.py) ---
main_pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X"]
pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X", "NZDUSD=X","AUDUSD=X", 'EURAUD=X', "EURNZD=X", "EURCHF=X" ]

# --- Config Macro (de terminal.py) ---
macro_data_config = {
    'USD': {'CPI': 'CPIAUCSL', 'GDP': 'GDP', 'PPI': 'PPIACO', 'Interest Rate': 'FEDFUNDS', 'Jobless Rate': 'UNRATE', 'GDP Growth': 'A191RL1Q225SBEA'},
    'EUR': {'CPI': 'CP0000EZ19M086NEST', 'GDP': 'CLVMNACSCAB1GQEA19', 'PPI': 'PPI_EA19', 'Interest Rate': 'ECBDFR', 'Jobless Rate': 'LRHUTTTTEZM156S', 'GDP Growth': 'NAEXKP01EZQ657S'},
    'GBP': {'CPI': 'CPALTT01GBM657N', 'GDP': 'NAEXKP01GBQ661S', 'PPI': 'PPIACOGBM086NEST', 'Interest Rate': 'BOEBASE', 'Jobless Rate': 'LRHUTTTTGBM156S', 'GDP Growth': 'NAEXKP01GBQ657S'},
    'JPY': {'CPI': 'CPALTT01JPM657N', 'GDP': 'CLVMNACSCAB1GQJP', 'PPI': 'PPIACOJPM086NEST', 'Interest Rate': 'IRSTCB01JPM156N', 'Jobless Rate': 'LRHUTTTTJPQ156S', 'GDP Growth': 'NAEXKP01JPQ657S'},
    'CAD': {'CPI': 'CPALTT01CAM657N', 'GDP': 'CLVMNACSCAB1GQCA', 'PPI': 'PPIACOCAM086NEST', 'Interest Rate': 'IRSTCB01CAM156N', 'Jobless Rate': 'LRHUTTTTCAQ156S', 'GDP Growth': 'NAEXKP01CAQ657S'}
}


# =============================================================================
# 1. FONCTIONS BACKEND (Modifiées pour Shiny : return au lieu de show())
# =============================================================================

@reactive.calc
def fetchMacroData_shiny():
    """Version Shiny de fetchMacroData de terminal.py"""
    results = []
    for devise, codes in macro_data_config.items():
        row = {'Devise': devise}
        for indicator, code in codes.items():
            try:
                serie = fred.get_series(code)
                if not serie.empty:
                    row[indicator] = serie.iloc[-1]
                else:
                    row[indicator] = np.nan
            except Exception:
                row[indicator] = np.nan
        results.append(row)
    macro_df = pd.DataFrame(results).set_index('Devise')
    return macro_df

def plotPrice_shiny(ticker: str):
    """Version Shiny de plotPrice de terminal.py, retourne fig"""
    data = yf.download(ticker, period="1y", interval="1h")
    data["SMA_200"] = data['Close'].rolling(window=200).mean()
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        name="Chandelier"
    ))
    fig_price.add_trace(go.Scatter(
        x=data.index, y=data["SMA_200"], mode="lines", 
        name="SMA 200", line=dict(color="green", dash="dot")
    ))
    fig_price.update_layout(
        title=f"{ticker} - Prix H1 (1 an)",
        xaxis_title="Date", yaxis_title="Prix",
        template="plotly_dark", legend_title="Légende",
        xaxis_rangeslider_visible=False
    )
    # MODIFICATION : Remplacer fig_price.show() par return fig_price
    return fig_price

def corrMatrix_shiny(tickers: list):
    """Version Shiny de corrMatrix, retourne fig matplotlib"""
    data = yf.download(tickers, period="10y", interval='1d')['Adj Close']
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    # MODIFICATION : Créer la figure mais ne pas l'afficher (pas de plt.show())
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr_matrix, 
        annot=True,     
        cmap='coolwarm',
        fmt='.2f',      
        linewidths=0.5,
        ax=ax
    )
    ax.set_title('Matrice de Corrélation des Rendements Journaliers (10 ans)')
    return fig

@reactive.calc
def seasonality_shiny(tickers: list):
    """Version Shiny de seasonality.py, retourne (fig, df)"""
    all_means = {}
    ranking_results = []
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="10y", interval="1mo").dropna()
            if data.empty: continue
            
            returns = data['Close'].pct_change().dropna()
            data['return'] = returns
            data["Month"] = data.index.month_name()
            mean_10y = data.groupby("Month")["return"].mean().reindex(month_order)
            all_means[ticker] = mean_10y

            best_month = mean_10y.idxmax()
            best_return = mean_10y.max()
            worst_month = mean_10y.idxmin()
            worst_return = mean_10y.min()
            
            ranking_results.append({
                "Ticker": ticker,
                "Best Month": best_month, "Best Return (%)": best_return * 100,
                "Worst Month": worst_month, "Worst Return (%)": worst_return * 100
            })
        except Exception:
            continue
            
    # Créer le DataFrame
    output_df = pd.DataFrame(ranking_results).set_index("Ticker")
    
    # Créer la Figure (Matplotlib)
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.style.use('dark_background')
    for ticker, mean in all_means.items():
        ax.plot(mean.index, mean.values * 100, marker='o', label=ticker)
    
    ax.set_title("Saisonnalité moyenne à 10 ans", color='white')
    ax.set_xlabel("Mois", color='white')
    ax.set_ylabel("Rendement moyen (%)", color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    
    # MODIFICATION : Retourner fig ET df
    return fig, output_df

@reactive.calc
def getVix_shiny():
    """Version Shiny de getVix, retourne fig"""
    data = yf.download("^VIX", period="3mo", interval="1h", progress=False)
    if data.empty: return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], mode="lines",
        name="VIX (Volatilité implicite du S&P 500)",
        line=dict(color="#00ff41", width=2)
    ))
    fig.update_layout(
        title="Indice VIX - Évolution sur 3 mois",
        xaxis_title="Date", yaxis_title="Valeur du VIX",
        template="plotly_dark", hovermode='x unified', height=500
    )
    return fig

@reactive.calc
def getPutCallRatio_shiny():
    """Version Shiny de getPutCallRatio, retourne (data, fig)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download('^CPCE', start=start_date, end=end_date, progress=False)
    if data.empty: return pd.DataFrame(), go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], mode='lines',
        name='CBOE Equity P/C Ratio', line=dict(color='#00ff41', width=2)
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Équilibre")
    fig.add_hline(y=0.7, line_dash="dot", line_color="#ff4136", annotation_text="Greed")
    fig.add_hline(y=1.2, line_dash="dot", line_color="#3D9970", annotation_text="Fear")
    fig.update_layout(
        title="CBOE Equity Put/Call Ratio (^CPCE)",
        xaxis_title="Date", yaxis_title="Put/Call Ratio",
        template="plotly_dark", height=500, hovermode='x unified'
    )
    return data, fig

def get_twitter_fx_sentiment_shiny(bearer_token: str, focus_pair: str = "EURUSD"):
    """Version Shiny de l'analyse Twitter, retourne fig"""
    if not bearer_token:
        print("Token Twitter manquant.")
        return go.Figure().update_layout(title="Veuillez fournir un Bearer Token", template="plotly_dark")
        
    try:
        client = tweepy.Client(bearer_token)
        analyzer = SentimentIntensityAnalyzer()
        query = f'("${focus_pair}" OR #{focus_pair} OR #Forex) lang:en -is:retweet'
        
        response = client.search_recent_tweets(query=query, max_results=100)
        tweets = response.data
        if not tweets:
            return go.Figure().update_layout(title="Aucun tweet trouvé", template="plotly_dark")

        sentiments = {"Positif": 0, "Négatif": 0, "Neutre": 0}
        
        def clean_tweet(text):
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+|#\w+|\$\w+', '', text)
            return text.strip()

        for tweet in tweets:
            score = analyzer.polarity_scores(clean_tweet(tweet.text))['compound']
            if score >= 0.05: sentiments["Positif"] += 1
            elif score <= -0.05: sentiments["Négatif"] += 1
            else: sentiments["Neutre"] += 1

        labels = list(sentiments.keys())
        values = list(sentiments.values())
        colors = ['#00ff41', '#ff4136', '#808080']

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, marker_colors=colors, hole=.3
        )])
        fig.update_layout(
            title=f"Sentiment Twitter pour '{focus_pair}' (sur {sum(values)} tweets)",
            template="plotly_dark", height=500, legend_title="Sentiment"
        )
        return fig
        
    except Exception as e:
        print(f"Erreur API Twitter : {e}")
        return go.Figure().update_layout(title=f"Erreur API: {e}", template="plotly_dark")


# =============================================================================
# 2. INTERFACE UTILISATEUR (UI)
# =============================================================================

app_ui = ui.page_navbar(
    ui.nav("Macro Dashboard",
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("### Données Macroéconomiques"),
                ui.p("Ce tableau affiche les dernières données macroéconomiques clés pour les principales zones monétaires, extraites de la FRED."),
                width=300
            ),
            ui.card(
                ui.output_data_frame("macro_table")
            )
        )
    ),
    ui.nav("Analyse de Prix",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Configuration"),
                ui.input_selectize("pair_select", "Choisir une paire", choices=pairs, selected="EURUSD=X"),
                ui.hr(),
                ui.p("Affiche le graphique en chandelier H1 sur 1 an avec une Moyenne Mobile Simple (SMA) de 200 périodes."),
                width=300
            ),
            ui.card(
                output_widget("price_plot") # Utiliser output_widget pour Plotly
            )
        )
    ),
    ui.nav("Analyse de Sentiment",
        ui.h3("Indicateurs de Sentiment de Marché"),
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("Volatilité (VIX)"),
                output_widget("vix_plot")
            ),
            ui.card(
                ui.card_header("CBOE Put/Call Ratio"),
                output_widget("put_call_plot")
            ),
            ui.card(
                ui.card_header("Sentiment Twitter (X)"),
                ui.input_select("twitter_pair", "Paire Forex", choices=main_pairs, selected="EURUSD=X"),
                ui.input_password("bearer_token", "Bearer Token (API X)"),
                ui.input_action_button("run_twitter", "Analyser", class_="btn-primary"),
                output_widget("twitter_plot")
            ),
            width="33.3%"
        )
    ),
    ui.nav("Corrélations",
        ui.card(
            ui.card_header("Matrice de Corrélation (10 ans, Rendements Journaliers)"),
            ui.output_plot("correlation_heatmap", height="700px") # output_plot pour Matplotlib
        )
    ),
    ui.nav("Saisonnalité",
        ui.card(
            ui.card_header("Performance Mensuelle Moyenne (10 ans)"),
            ui.output_plot("seasonality_plot", height="500px")
        ),
        ui.card(
            ui.card_header("Meilleurs & Pires Mois (Résumé)"),
            ui.output_data_frame("seasonality_table")
        )
    ),
    title="OptiWealth Terminal",
    inverse=True, # Thème sombre
    header=ui.tags.style(
        """
        .card { height: 100%; }
        .card-body { height: calc(100% - 48px); overflow-y: auto; }
        """
    )
)

# =============================================================================
# 3. LOGIQUE SERVEUR (Server)
# =============================================================================

def server(input: shiny.Inputs, output: shiny.Outputs, session: shiny.Session):

    # --- Onglet Macro ---
    @output
    @render.data_frame
    def macro_table():
        df = fetchMacroData_shiny()
        return render.DataGrid(df.reset_index(), filters=True)

    # --- Onglet Prix ---
    @output
    @render_widget # Utiliser render_widget pour Plotly
    def price_plot():
        return plotPrice_shiny(input.pair_select())

    # --- Onglet Sentiment ---
    @output
    @render_widget
    def vix_plot():
        return getVix_shiny()

    @output
    @render_widget
    def put_call_plot():
        # La fonction retourne (data, fig), on prend la fig
        data, fig = getPutCallRatio_shiny()
        return fig
    
    @output
    @render_widget
    @reactive.event(input.run_twitter) # Se déclenche au clic
    def twitter_plot():
        token = input.bearer_token()
        pair = input.twitter_pair()
        return get_twitter_fx_sentiment_shiny(token, pair)

    # --- Onglet Corrélations ---
    @output
    @render.plot(alt="Matrice de corrélation") # render.plot pour Matplotlib
    def correlation_heatmap():
        fig = corrMatrix_shiny(pairs)
        return fig

    # --- Onglet Saisonnalité ---
    # Utiliser un reactive.calc pour n'exécuter l'analyse qu'une fois
    seasonal_data = reactive.calc(lambda: seasonality_shiny(pairs))

    @output
    @render.plot(alt="Graphique de saisonnalité")
    def seasonality_plot():
        fig, df = seasonal_data() # Récupère la (fig, df)
        return fig
    
    @output
    @render.data_frame
    def seasonality_table():
        fig, df = seasonal_data() # Récupère la (fig, df)
        return render.DataGrid(df.reset_index(), filters=True)


# =============================================================================
# 4. Lancement de l'application
# =============================================================================

app = App(app_ui, server)