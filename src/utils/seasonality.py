import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def analyze_seasonality(ticker: str):
    # Téléchargement des données sur 10 ans
    data = yf.download(ticker, period="10y", interval="1mo")['Adj Close'].dropna()

    # Calcul des rendements mensuels
    returns = data.pct_change().dropna()

    # Ajout des colonnes d'année et de mois
    df = pd.DataFrame({"Return": returns})
    df["Year"] = df.index.year
    df["Month"] = df.index.month_name()

    # Rendements moyens mensuels sur 10 ans
    mean_10y = df.groupby("Month")["Return"].mean().sort_index(key=lambda x: pd.to_datetime(x, format="%B"))

    # Rendements moyens mensuels sur 5 ans
    recent_df = df[df["Year"] >= df["Year"].max() - 5]
    mean_5y = recent_df.groupby("Month")["Return"].mean().sort_index(key=lambda x: pd.to_datetime(x, format="%B"))

    # Tableau des meilleurs et pires mois sur 10 ans
    ranking = mean_10y.sort_values(ascending=False).reset_index()
    ranking.columns = ["Month", "Average_Return_10Y"]

    print("📊 Classement des mois (sur 10 ans) :\n")
    print(ranking)

    # Visualisation
    plt.figure(figsize=(12, 6))
    months = mean_10y.index
    plt.plot(months, mean_10y.values, label="Moyenne 10 ans", marker='o')
    plt.plot(months, mean_5y.values, label="Moyenne 5 ans", marker='o', linestyle="--")
    plt.title(f"Saisonnalité moyenne de {ticker}")
    plt.xlabel("Mois")
    plt.ylabel("Rendement moyen (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ranking

analyze_seasonality("AAPL")