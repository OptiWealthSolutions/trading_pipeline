import yfinance as yf
import pandas  as pd
import numpy as np
import fpdf 
import seaborn as sns
from fredapi import Fred
import plotly.graph_objects as go  
import matplotlib.pyplot as plt 
from polygon import RESTClient
import time
from config import polygon_api_key
client = RESTClient(polygon_api_key)

def getCOT():
    return 

def getPutCallRatio():
    underlying_stock = 'SP500'
    start_date = '2025-09-01'
    active_contracts = []
    for option in client.list_options_contracts(underlying_ticker=underlying_stock, expiration_date_gte=start_date, limit=1000):
        active_contracts.append(option)
    expired_contracts = []
    for option in client.list_options_contracts(underlying_ticker=underlying_stock, expiration_date_gte=start_date, limit=1000, expired=True):
        expired_contracts.append(option)
    options_contracts = active_contracts + expired_contracts
    df_options_contracts_dim = pd.DataFrame(options_contracts)
    
    df_list = []
    for contract in options_contracts:
        aggs = client.list_aggs(
            ticker=contract.ticker,
            multiplier=1,
            timespan='day',
            from_=start_date,
            to='2025-10-25',
            limit=50000
        )
        df_list.append(pd.DataFrame(aggs).assign(ticker=contract.ticker))
    
    # Combine DataFrames at the End
    df = pd.concat(df_list, ignore_index=True)
    
    # Make timestamp Human Readable
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    elif 't' in df.columns:
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', errors='coerce')
    else:
        raise KeyError("Aucune colonne de temps ('timestamp' ou 't') trouvée dans les données Polygon.")
    
    df = df_options_contracts_dim.merge(df, on='ticker')
    
    df['call_volume'] = np.where(df['contract_type'] == 'call', df.get('volume', 0), 0)
    df['put_volume'] = np.where(df['contract_type'] == 'put', df.get('volume', 0), 0)
    
    df_grouped = df.groupby('timestamp')[['put_volume','call_volume']].sum()
    df_grouped['put_call_ratio'] = df_grouped['put_volume'] / df_grouped['call_volume']
    
    # Correct plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,9))
    plt.plot(df_grouped.index, df_grouped['put_call_ratio'], label='Put/Call Ratio', color='green')
    plt.xlabel("Date")
    plt.ylabel("Put/Call Ratio")
    plt.title(f"Put/Call Ratio - {underlying_stock}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    
    return df_grouped

getPutCallRatio()

import tweepy
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re

def get_twitter_fx_sentiment(bearer_token: str, focus_pair: str = "EURUSD"):
    if not bearer_token:
        print("Erreur : Le 'bearer_token' est manquant. Impossible de se connecter à l'API X.")
        return None

    try:
        client = tweepy.Client(bearer_token)
        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Erreur lors de l'initialisation des clients : {e}")
        return None

    # Création d'une requête de recherche ciblée
    # (Recherche la paire + termes généraux, en anglais, sans retweets)
    query = f'("${focus_pair}" OR #{focus_pair} OR #Forex OR #FXTrading) lang:en -is:retweet'
    
    sentiment_scores = []

    # --- 2. Récupération des Tweets ---
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=100,  # Max 100 par l'API (pour un test gratuit)
            tweet_fields=["text", "created_at"]
        )
        
        if not response.data:
            print("Aucun tweet trouvé pour la requête : {query}")
            return None
        
        tweets = response.data

    except Exception as e:
        print(f"Erreur lors de la récupération des tweets : {e}")
        return None

    # --- 3. Analyse de Sentiment (VADER) ---
    sentiments = {"Positif": 0, "Négatif": 0, "Neutre": 0}

    def clean_tweet(text):
        text = re.sub(r'http\S+', '', text)  # Enlève les URLs
        text = re.sub(r'@\w+', '', text)     # Enlève les mentions
        text = re.sub(r'#\w+', '', text)     # Enlève les hashtags (le texte seul)
        text = re.sub(r'\$\w+', '', text)    # Enlève les cashtags
        return text.strip()

    for tweet in tweets:
        cleaned_text = clean_tweet(tweet.text)
        
        # Obtention du score VADER
        # 'compound' est un score global de -1 (très nég) à +1 (très pos)
        score = analyzer.polarity_scores(cleaned_text)['compound']
        
        # Classification du sentiment
        if score >= 0.05:
            sentiments["Positif"] += 1
        elif score <= -0.05:
            sentiments["Négatif"] += 1
        else:
            sentiments["Neutre"] += 1

    # --- 4. Création du Diagramme Circulaire (Plotly) ---
    labels = list(sentiments.keys())
    values = list(sentiments.values())
    
    # Définition des couleurs pour le sentiment
    colors = ['#00ff41', '#ff4136', '#808080'] # Positif(vert), Négatif(rouge), Neutre(gris)

    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        pull=[0.05 if l == "Positif" or l == "Négatif" else 0 for l in labels], # Décolle les parts
        marker_colors=colors,
        hole=.3 # Effet "Donut"
    )])

    fig.update_layout(
        title=f"Sentiment Twitter (FX) pour '{focus_pair}' (sur {sum(values)} tweets)",
        template="plotly_dark",
        legend_title="Sentiment"
    )

    return fig
