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
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
# Chargez les clés depuis les variables d'environnement (plus sûr)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "e16626c91fa2b1af27704a783939bf72")
TWITTER_BEARER_TOKEN_ENV = os.environ.get("TWITTER_BEARER_TOKEN")

try:
    fred = Fred(api_key=FRED_API_KEY)
except Exception as e:
    print(f"Erreur de connexion à FRED: {e}. Vérifiez votre clé API.")
    # L'application plantera, ce qui est normal si FRED n'est pas accessible.

# --- Constantes ---
main_pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X"]
pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X", "NZDUSD=X", "AUDUSD=X", 'EURAUD=X', "EURNZD=X", "EURCHF=X"]
commodities = {"Or": "GC=F", "Pétrole (WTI)": "CL=F"}

macro_data_config = {
    'USD': {'CPI': 'CPIAUCSL', 'GDP': 'GDP', 'PPI': 'PPIACO', 'Interest Rate': 'FEDFUNDS', 'Jobless Rate': 'UNRATE', 'GDP Growth': 'A191RL1Q225SBEA'},
    'EUR': {'CPI': 'CP0000EZ19M086NEST', 'GDP': 'CLVMNACSCAB1GQEA19', 'PPI': 'PPI_EA19', 'Interest Rate': 'ECBDFR', 'Jobless Rate': 'LRHUTTTTEZM156S', 'GDP Growth': 'NAEXKP01EZQ657S'},
    'GBP': {'CPI': 'CPALTT01GBM657N', 'GDP': 'NAEXKP01GBQ661S', 'PPI': 'PPIACOGBM086NEST', 'Interest Rate': 'BOEBASE', 'Jobless Rate': 'LRHUTTTTGBM156S', 'GDP Growth': 'NAEXKP01GBQ657S'},
    'JPY': {'CPI': 'CPALTT01JPM657N', 'GDP': 'CLVMNACSCAB1GQJP', 'PPI': 'PPIACOJPM086NEST', 'Interest Rate': 'IRSTCB01JPM156N', 'Jobless Rate': 'LRHUTTTTJPQ156S', 'GDP Growth': 'NAEXKP01JPQ657S'},
    'CAD': {'CPI': 'CPALTT01CAM657N', 'GDP': 'CLVMNACSCAB1GQCA', 'PPI': 'PPIACOCAM086NEST', 'Interest Rate': 'IRSTCB01CAM156N', 'Jobless Rate': 'LRHUTTTTCAQ156S', 'GDP Growth': 'NAEXKP01CAQ657S'}
}

# =============================================================================
# STYLE CSS PRO TERMINAL
# =============================================================================
custom_css = """
<style>
:root {
    --bg-primary: #0d1117;       /* Noir profond (GitHub) */
    --bg-secondary: #161a25;     /* Panneaux (Bleu très sombre) */
    --bg-card: #161a25;
    --accent-cyan: #00d4ff;      /* Accent principal */
    --accent-gold: #d4af37;      /* Accent secondaire */
    --accent-red: #ff4757;
    --accent-green: #00ff88;
    --text-primary: #e6eef6;     /* Texte principal (Blanc cassé) */
    --text-secondary: #7d8da1;   /* Texte muet (Gris-bleu) */
    --border-color: rgba(255, 255, 255, 0.1);
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

body {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-family) !important;
    font-size: 14px;
}

/* --- Navbar --- */
.navbar {
    background: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border-color) !important;
    box-shadow: none !important;
}
.navbar-brand {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
.nav-link {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border-bottom: 3px solid transparent !important;
}
.nav-link.active, .nav-link:hover {
    color: var(--accent-cyan) !important;
    border-bottom: 3px solid var(--accent-cyan) !important;
}

/* --- Cartes et Panneaux --- */
.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    margin-bottom: 20px !important;
}
.card-header {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, #1c2230 100%) !important;
    border-bottom: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1rem;
    padding: 0.75rem 1.25rem !important;
    border-radius: 8px 8px 0 0 !important;
}
.card-body {
    padding: 1.25rem !important;
}

/* --- Sidebar --- */
.sidebar {
    background: var(--bg-primary) !important;
    border-right: 1px solid var(--border-color) !important;
    padding: 1.5rem !important;
}

/* --- Formulaires --- */
.form-control, .form-select {
    background-color: #0d1117 !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
.form-control:focus, .form-select:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
    background-color: #0d1117 !important;
}

/* --- Boutons --- */
.btn-primary {
    background: linear-gradient(135deg, var(--accent-cyan) 0%, #009bcc 100%) !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    color: #0d1117 !important;
    box-shadow: 0 2px 8px rgba(0, 212, 255, 0.2) !important;
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3) !important;
}

/* --- Texte --- */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600;
}
p, span, div, label {
    color: var(--text-secondary); /* Texte muet par défaut */
}
.text-primary { color: var(--text-primary) !important; } /* Classe pour forcer le texte blanc */

/* --- Value Boxes --- */
.value-box {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    text-align: left;
}
.value-box-title {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
}
.value-box-value {
    color: var(--text-primary);
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}
.value-box-subtitle {
    color: var(--accent-gold);
    font-size: 0.9rem;
    margin-top: 0.25rem;
}
.value-box-subtitle.positive { color: var(--accent-green); }
.value-box-subtitle.negative { color: var(--accent-red); }

/* --- DataGrid --- */
.shiny-data-grid {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
}
.shiny-data-grid .header {
    background: #1c2230 !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border-color) !important;
}
.shiny-data-grid .cell {
    color: var(--text-secondary) !important;
    border-color: var(--border-color) !important;
}
.shiny-data-grid .row.odd { background-color: var(--bg-card) !important; }
.shiny-data-grid .row.even { background-color: #1a1f2c !important; }
.shiny-data-grid .row:hover { background-color: rgba(0, 212, 255, 0.1) !important; }

hr { border-color: var(--border-color) !important; opacity: 0.5 !important; }
</style>
"""

# =============================================================================
# FONCTIONS BACKEND (Nouvelles fonctions + Style Pro)
# =============================================================================

# --- NOUVELLES FONCTIONS POUR LE DASHBOARD ---

@reactive.calc
def get_performance_heatmap_shiny():
    """Crée une heatmap de performance pour les paires (1J, 1S, 1M, 3M)"""
    try:
        data = yf.download(pairs, period="4mo", progress=False)['Close']
        if data.empty:
            raise Exception("Aucune donnée de performance reçue de yfinance")

        periods = {
            "1 Jour": 1,
            "1 Semaine": 5,
            "1 Mois": 21,
            "3 Mois": 63
        }
        perf_df = pd.DataFrame(index=pairs)
        
        for name, period in periods.items():
            perf_df[name] = (data.pct_change(period).iloc[-1] * 100)
            
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#161a25') # Couleur de la carte
        ax.set_facecolor('#161a25')
        
        sns.heatmap(
            perf_df,
            annot=True,
            cmap="vlag", # Rouge (-1) / Blanc (0) / Bleu (+1)
            fmt='.2f',
            linewidths=2,
            linecolor=ax.get_facecolor(),
            ax=ax,
            cbar=False,
            annot_kws={"color": "white", "size": 10}
        )
        
        ax.set_title('Heatmap de Performance (FX)', color='var(--text-primary)', fontsize=14, pad=15)
        plt.xticks(rotation=0, color='var(--text-secondary)', fontsize=10)
        plt.yticks(rotation=0, color='var(--text-primary)', fontsize=10)
        fig.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Erreur Heatmap: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#161a25')
        ax.set_facecolor('#161a25')
        ax.text(0.5, 0.5, f"Erreur de chargement des données:\n{e}", ha='center', va='center', color='var(--accent-red)')
        return fig

@reactive.calc
def get_yield_curve_shiny():
    """Crée un graphique de la courbe des taux (10Y, 2Y) et du spread"""
    try:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        df_10y = fred.get_series('DGS10', observation_start=start_date)
        df_2y = fred.get_series('DGS2', observation_start=start_date)
        
        df = pd.DataFrame({'US 10Y': df_10y, 'US 2Y': df_2y}).dropna()
        df['Spread (10Y-2Y)'] = df['US 10Y'] - df['US 2Y']
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot 1: Taux
        fig.add_trace(go.Scatter(
            x=df.index, y=df['US 10Y'], name='Taux US 10 ans',
            line=dict(color='var(--accent-cyan)', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['US 2Y'], name='Taux US 2 ans',
            line=dict(color='var(--accent-gold)', width=2, dash='dash')
        ), row=1, col=1)
        
        # Plot 2: Spread
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Spread (10Y-2Y)'], name='Spread',
            line=dict(color='var(--text-primary)', width=2),
            fill='tozeroy', fillcolor='rgba(230, 238, 246, 0.1)'
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color='var(--accent-red)', row=2, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#161a25', plot_bgcolor='#161a25',
            font_color='var(--text-primary)', height=400,
            hovermode='x unified', margin=dict(l=40, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Taux (%)", row=1, col=1)
        fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
        
        return fig

    except Exception as e:
        print(f"Erreur Yield Curve: {e}")
        return go.Figure().update_layout(
            title="Erreur de chargement des données de la FRED",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--accent-red)'
        )

@reactive.calc
def get_commodities_shiny():
    """Crée un graphique pour l'Or et le Pétrole"""
    try:
        data = yf.download(list(commodities.values()), period="1y", progress=False)['Close']
        data = data.rename(columns={"GC=F": "Or", "CL=F": "Pétrole (WTI)"})
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Or'], name='Or (GC=F)',
            line=dict(color='var(--accent-gold)', width=2),
            fill='tozeroy', fillcolor='rgba(212, 175, 55, 0.1)'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Pétrole (WTI)'], name='Pétrole (CL=F)',
            line=dict(color='var(--text-secondary)', width=2),
            fill='tozeroy', fillcolor='rgba(125, 141, 161, 0.1)'
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#161a25', plot_bgcolor='#161a25',
            font_color='var(--text-primary)', height=500,
            hovermode='x unified', showlegend=False, margin=dict(l=40, r=20, t=40, b=20)
        )
        fig.update_yaxes(title_text="Prix Or (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Prix Pétrole (USD)", row=2, col=1)
        
        return fig

    except Exception as e:
        print(f"Erreur Commodities: {e}")
        return go.Figure().update_layout(
            title="Erreur de chargement des données Matières Premières",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--accent-red)'
        )

# --- FONCTIONS EXISTANTES (STYLE MIS À JOUR) ---

@reactive.calc
def fetchMacroData_shiny():
    """Récupère les données macroéconomiques (inchangé, backend)"""
    results = []
    for devise, codes in macro_data_config.items():
        row = {'Devise': devise}
        for indicator, code in codes.items():
            try:
                serie = fred.get_series(code)
                row[indicator] = serie.iloc[-1] if not serie.empty else np.nan
            except Exception:
                row[indicator] = np.nan
        results.append(row)
    return pd.DataFrame(results).set_index('Devise')

@reactive.calc
def getDashboardMetrics_shiny():
    """Récupère les métriques clés pour le dashboard (inchangé, backend)"""
    metrics = {}
    try:
        unrate = fred.get_series('UNRATE')
        metrics['unemployment'] = unrate.iloc[-1] if not unrate.empty else None
        gdp_growth = fred.get_series('A191RL1Q225SBEA')
        metrics['gdp_growth'] = gdp_growth.iloc[-1] if not gdp_growth.empty else None
        fedfunds = fred.get_series('FEDFUNDS')
        metrics['fed_rate'] = fedfunds.iloc[-1] if not fedfunds.empty else None
        cpi = fred.get_series('CPIAUCSL')
        if not cpi.empty and len(cpi) >= 13:
            metrics['inflation'] = ((cpi.iloc[-1] / cpi.iloc[-13]) - 1) * 100
        else:
            metrics['inflation'] = None
    except Exception as e:
        print(f"Erreur Métriques Dashboard: {e}")
    return metrics

def plotPrice_shiny(ticker: str):
    """Graphique de prix (Style Pro)"""
    try:
        data = yf.download(ticker, period="1y", interval="1h", progress=False)
        if data.empty:
            raise Exception(f"Aucune donnée yfinance pour {ticker}")
            
        data["SMA_200"] = data['Close'].rolling(window=200).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
            name="Prix",
            increasing_line_color='var(--accent-green)',
            decreasing_line_color='var(--accent-red)'
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200",
            line=dict(color='var(--accent-cyan)', width=2, dash="dot")
        ))
        
        fig.update_layout(
            title=f"{ticker} - Analyse Technique (1 an, H1)",
            template="plotly_dark", paper_bgcolor='#161a25', plot_bgcolor='#161a25',
            font_color='var(--text-primary)', height=550,
            xaxis_rangeslider_visible=False, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=20)
        )
        return fig
        
    except Exception as e:
        print(f"Erreur plotPrice: {e}")
        return go.Figure().update_layout(
            title=f"Erreur de chargement: {e}",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--accent-red)'
        )

def corrMatrix_shiny(tickers: list):
    """Matrice de corrélation (Style Pro)"""
    try:
        data = yf.download(tickers, period="5y", interval='1d', progress=False)['Close']
        if data.empty:
            raise Exception("Aucune donnée de corrélation reçue")
            
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor('#161a25')
        ax.set_facecolor('#161a25')
        
        sns.heatmap(
            corr_matrix, annot=True, cmap="vlag", fmt='.2f',
            linewidths=2, linecolor=ax.get_facecolor(), ax=ax,
            cbar=False, vmin=-1, vmax=1, center=0,
            annot_kws={"color": "white", "size": 10}
        )
        
        ax.set_title('Matrice de Corrélation (5 ans)', color='var(--text-primary)', fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', color='var(--text-secondary)', fontsize=10)
        plt.yticks(rotation=0, color='var(--text-primary)', fontsize=10)
        fig.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Erreur corrMatrix: {e}")
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor('#161a25')
        ax.set_facecolor('#161a25')
        ax.text(0.5, 0.5, f"Erreur de chargement des données:\n{e}", ha='center', va='center', color='var(--accent-red)')
        return fig

def seasonality_shiny(tickers: list):
    """Analyse de saisonnalité (Style Pro)"""
    all_means = {}
    ranking_results = []
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="10y", interval="1mo", progress=False).dropna()
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
        except Exception as e:
            print(f"Erreur Saisonnalité pour {ticker}: {e}")
            continue
    
    output_df = pd.DataFrame(ranking_results).set_index("Ticker")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#161a25')
    ax.set_facecolor('#161a25')
    
    colors_palette = ['#00d4ff', '#d4af37', '#00ff88', '#ff6b6b', '#a29bfe', 
                      '#fd79a8', '#00cec9', '#fdcb6e', '#e17055']
    
    for idx, (ticker, mean) in enumerate(all_means.items()):
        color = colors_palette[idx % len(colors_palette)]
        ax.plot(mean.index, mean.values * 100, marker='o', label=ticker,
                linewidth=2, markersize=6, color=color)
    
    ax.set_title("Saisonnalité Moyenne (10 ans)", fontsize=14, color='var(--text-primary)', pad=15)
    ax.set_xlabel("Mois", fontsize=10, color='var(--text-secondary)')
    ax.set_ylabel("Rendement Moyen (%)", fontsize=10, color='var(--text-secondary)')
    
    ax.legend(facecolor='#1c2230', edgecolor='var(--border-color)', labelcolor='var(--text-primary)',
              framealpha=1, loc='best', fontsize=9)
    ax.grid(True, color='var(--border-color)', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='var(--text-secondary)', linestyle='-', linewidth=1, alpha=0.7)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color='var(--text-secondary)', fontsize=10)
    plt.setp(ax.get_yticklabels(), color='var(--text-secondary)', fontsize=10)
    
    fig.tight_layout()
    return fig, output_df

def getVix_shiny():
    """Indice VIX (Style Pro)"""
    try:
        data = yf.download("^VIX", period="3mo", interval="1h", progress=False)
        if data.empty: raise Exception("Aucune donnée VIX")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], name="VIX",
            line=dict(color='var(--accent-cyan)', width=2),
            fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(0, 255, 136, 0.1)", layer="below", line_width=0,
                      annotation_text="Faible Volatilité", annotation_position="top left")
        fig.add_hrect(y0=20, y1=30, fillcolor="rgba(212, 175, 55, 0.1)", layer="below", line_width=0)
        fig.add_hrect(y0=30, y1=100, fillcolor="rgba(255, 71, 87, 0.1)", layer="below", line_width=0,
                      annotation_text="Haute Volatilité", annotation_position="top left")
        
        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#161a25', plot_bgcolor='#161a25',
            font_color='var(--text-primary)', height=400,
            hovermode='x unified', margin=dict(l=40, r=20, t=40, b=20),
            showlegend=False, yaxis_title="VIX"
        )
        return fig
        
    except Exception as e:
        print(f"Erreur VIX: {e}")
        return go.Figure().update_layout(
            title="Erreur de chargement des données VIX",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--accent-red)'
        )

def getPutCallRatio_shiny():
    """Ratio Put/Call (Style Pro)"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download('^CPCE', start=start_date, end=end_date, progress=False)
        if data.empty: raise Exception("Aucune donnée P/C Ratio")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], name='P/C Ratio',
            line=dict(color='var(--accent-cyan)', width=2),
            fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="var(--text-secondary)",
                      annotation_text="Équilibre (1.0)", annotation_position="right")
        fig.add_hline(y=0.7, line_dash="dot", line_color="var(--accent-green)",
                      annotation_text="Avidité (Calls > Puts)", annotation_position="bottom right")
        fig.add_hline(y=1.2, line_dash="dot", line_color="var(--accent-red)",
                      annotation_text="Peur (Puts > Calls)", annotation_position="top right")
        
        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#161a25', plot_bgcolor='#161a25',
            font_color='var(--text-primary)', height=400,
            hovermode='x unified', margin=dict(l=40, r=20, t=40, b=20),
            showlegend=False, yaxis_title="Put/Call Ratio"
        )
        return fig
        
    except Exception as e:
        print(f"Erreur P/C Ratio: {e}")
        return go.Figure().update_layout(
            title="Erreur de chargement des données P/C Ratio",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--accent-red)'
        )

def get_twitter_fx_sentiment_shiny(bearer_token: str, focus_pair: str = "EURUSD"):
    """Analyse de sentiment Twitter (Style Pro)"""
    if not bearer_token:
        return go.Figure().update_layout(
            title="Veuillez fournir un Bearer Token API X dans le panneau",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--text-secondary)'
        )
    
    try:
        client = tweepy.Client(bearer_token)
        analyzer = SentimentIntensityAnalyzer()
        query = f'("${focus_pair}" OR #{focus_pair} OR #Forex) lang:en -is:retweet'
        
        response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["created_at"])
        tweets = response.data
        
        if not tweets:
            return go.Figure().update_layout(
                title=f"Aucun tweet récent trouvé pour {focus_pair}",
                template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--text-secondary)'
            )
        
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
        colors = ['var(--accent-green)', 'var(--accent-red)', 'var(--text-secondary)']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, marker_colors=colors,
            hole=.4, textinfo='label+percent',
            textfont=dict(size=14, color='var(--text-primary)'),
            hovertemplate='<b>%{label}</b><br>Tweets: %{value}<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f"Sentiment Twitter: {focus_pair} (sur {sum(values)} tweets)",
            template="plotly_dark", paper_bgcolor='#161a25',
            height=400, showlegend=False,
            font_color='var(--text-primary)',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig
        
    except Exception as e:
        print(f"Erreur Twitter API: {e}")
        return go.Figure().update_layout(
            title=f"Erreur API X: Vérifiez votre Bearer Token ou vos droits d'accès.",
            template="plotly_dark", paper_bgcolor='#161a25', font_color='var(--accent-red)'
        )

# =============================================================================
# INTERFACE UTILISATEUR (UI) - Refonte Pro
# =============================================================================

# Fonction d'aide pour les Value Boxes
def create_value_box(title: str, value_id: str, subtitle_id: str):
    return ui.div(
        ui.div(title, class_="value-box-title"),
        
        # --- CORRECTION ICI ---
        # On enveloppe output_text dans un div pour lui appliquer la classe
        ui.div(
            ui.output_text(value_id), 
            class_="value-box-value"
        ),
        
        # --- CORRECTION ICI ---
        # On enveloppe output_ui dans un div pour lui appliquer la classe
        ui.div(
            ui.output_ui(subtitle_id),
            class_="value-box-subtitle"
        ),
        # --- FIN DE LA CORRECTION ---
        
        class_="value-box"
    )

app_ui = ui.page_navbar(
    ui.tags.head(ui.tags.style(custom_css)),
    
    # --- NOUVEL ONGLET DASHBOARD ---
    ui.nav_panel(
        "🌍 Dashboard",
        ui.layout_column_wrap(
            create_value_box("Taux FED", "fed_funds_rate", "fed_funds_subtitle"),
            create_value_box("Inflation US (YoY)", "inflation_rate", "inflation_subtitle"),
            create_value_box("Croissance PIB US (QoQ)", "gdp_growth_rate", "gdp_subtitle"),
            create_value_box("Chômage US", "unemployment_rate", "unemployment_subtitle"),
            width="25%"
        ),
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("Heatmap de Performance (FX)"),
                ui.output_plot("performance_heatmap", height="400px")
            ),
            ui.card(
                ui.card_header("Courbe des Taux US (10Y-2Y)"),
                output_widget("yield_curve_plot")
            ),
            width="50%"
        )
    ),
    
    # --- NOUVEL ONGLET MARCHÉS (regroupé) ---
    ui.nav_panel(
        "💹 Marchés",
        ui.navset_card_tab(
            ui.nav_panel(
                "Analyse de Prix (FX)",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h4("Configuration", class_="text-primary"),
                        ui.input_selectize("pair_select", "Sélectionner une paire",
                                         choices=pairs, selected="EURUSD=X"),
                        ui.hr(),
                        ui.p("Analyse H1 sur 1 an avec SMA 200.", class_="small"),
                        width=300
                    ),
                    output_widget("price_plot")
                )
            ),
            ui.nav_panel(
                "Matières Premières",
                output_widget("commodities_plot")
            ),
            ui.nav_panel(
                "Volatilité",
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Indice VIX"),
                        output_widget("vix_plot")
                    ),
                    ui.card(
                        ui.card_header("Put/Call Ratio (CPCE)"),
                        output_widget("put_call_plot")
                    ),
                    width="50%"
                )
            )
        )
    ),
    
    # --- NOUVEL ONGLET ANALYSE (regroupé) ---
    ui.nav_panel(
        "🔬 Analyse",
        ui.navset_card_tab(
            ui.nav_panel(
                "Corrélations",
                ui.card(
                    ui.card_header("Matrice de Corrélation (5 ans)"),
                    ui.output_plot("correlation_heatmap", height="700px")
                )
            ),
            ui.nav_panel(
                "Saisonnalité",
                ui.card(
                    ui.card_header("Performance Mensuelle Moyenne (10 ans)"),
                    ui.output_plot("seasonality_plot", height="500px")
                ),
                ui.card(
                    ui.card_header("Classement: Meilleurs & Pires Mois"),
                    ui.output_data_frame("seasonality_table")
                )
            )
        )
    ),
    
    # --- ONGLET SENTIMENT ---
    ui.nav_panel(
        "📰 Sentiment",
        ui.card(
            ui.card_header("Sentiment Twitter (X)"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("twitter_pair", "Paire Forex",
                                   choices=main_pairs, selected="EURUSD=X"),
                    ui.input_password("bearer_token", "Bearer Token (API X)"),
                    ui.input_action_button("run_twitter", "Analyser", class_="btn-primary w-100"),
                    ui.p("Un token API (Free Tier) est requis.", class_="small mt-3"),
                    width=300
                ),
                output_widget("twitter_plot")
            )
        )
    ),
    
    # --- ONGLET DONNÉES MACRO ---
    ui.nav_panel(
        "📊 Données Macro",
        ui.card(
            ui.card_header("Tableau des Données Macro (FRED)"),
            ui.output_data_frame("macro_table")
        )
    ),
    
    title="OptiWealth Terminal",
    inverse=False,
    fillable=True
)

# =============================================================================
# LOGIQUE SERVEUR (Refonte)
# =============================================================================

def server(input: shiny.Inputs, output: shiny.Outputs, session: shiny.Session):

    # --- Cache Réactif pour les Métriques ---
    metrics = reactive.calc(getDashboardMetrics_shiny)
    
    def render_metric_subtitle(metric_value):
        cls = "positive" if metric_value > 0 else "negative" if metric_value < 0 else ""
        return ui.span(f"{metric_value:+.2f}%" if metric_value is not None else "N/A", class_=cls)

    # --- Rendu du Dashboard ---
    @output
    @render.text
    def fed_funds_rate():
        val = metrics().get('fed_rate')
        return f"{val:.2f}%" if val is not None else "N/A"
    @output
    @render.ui
    def fed_funds_subtitle():
        return ui.span("Taux Effectif", class_="") # Pas de changement

    @output
    @render.text
    def inflation_rate():
        val = metrics().get('inflation')
        return f"{val:.2f}%" if val is not None else "N/A"
    @output
    @render.ui
    def inflation_subtitle():
        val = metrics().get('inflation')
        return render_metric_subtitle(val if val is not None else 0)

    @output
    @render.text
    def gdp_growth_rate():
        val = metrics().get('gdp_growth')
        return f"{val:.2f}%" if val is not None else "N/A"
    @output
    @render.ui
    def gdp_subtitle():
        val = metrics().get('gdp_growth')
        return render_metric_subtitle(val if val is not None else 0)

    @output
    @render.text
    def unemployment_rate():
        val = metrics().get('unemployment')
        return f"{val:.1f}%" if val is not None else "N/A"
    @output
    @render.ui
    def unemployment_subtitle():
        return ui.span("Dernier mois", class_="") # Pas de changement

    @output
    @render.plot(alt="Heatmap de Performance")
    def performance_heatmap():
        return get_performance_heatmap_shiny()

    @output
    @render_widget
    def yield_curve_plot():
        return get_yield_curve_shiny()

    # --- Rendu des Marchés ---
    @output
    @render_widget
    def price_plot():
        return plotPrice_shiny(input.pair_select())

    @output
    @render_widget
    def commodities_plot():
        return get_commodities_shiny()

    @output
    @render_widget
    def vix_plot():
        return getVix_shiny()

    @output
    @render_widget
    def put_call_plot():
        data, fig = getPutCallRatio_shiny()
        return fig

    # --- Rendu de l'Analyse ---
    @output
    @render.plot(alt="Matrice de corrélation")
    def correlation_heatmap():
        return corrMatrix_shiny(pairs)

    seasonal_data = reactive.calc(lambda: seasonality_shiny(pairs))

    @output
    @render.plot(alt="Graphique de saisonnalité")
    def seasonality_plot():
        fig, df = seasonal_data()
        return fig
    
    @output
    @render.data_frame
    def seasonality_table():
        fig, df = seasonal_data()
        return render.DataGrid(df.reset_index().round(2), filters=True, height="400px")

    # --- Rendu du Sentiment ---
    twitter_token = reactive.Value(TWITTER_BEARER_TOKEN_ENV)

    @reactive.Effect
    @reactive.event(input.bearer_token)
    def _set_token():
        if input.bearer_token():
            twitter_token.set(input.bearer_token())

    @output
    @render_widget
    @reactive.event(input.run_twitter, ignore_none=False)
    def twitter_plot():
        token = twitter_token.get()
        return get_twitter_fx_sentiment_shiny(token, input.twitter_pair())

    # --- Rendu des Données Macro ---
    @output
    @render.data_frame
    def macro_table():
        df = fetchMacroData_shiny()
        return render.DataGrid(df.reset_index().round(2), filters=True, height="600px")

# =============================================================================
# LANCEMENT DE L'APPLICATION
# =============================================================================

app = App(app_ui, server)