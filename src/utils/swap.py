from fredapi import Fred
import matplotlib.pyplot as plt
import pandas as pd

# Initialisation de l'API FRED
fred = Fred(api_key="e16626c91fa2b1af27704a783939bf72")

# Dictionnaire des taux de swap (principalement taux OIS et swaps de référence)
# Note: FRED a une disponibilité limitée pour les taux de swap
tickers = {
    # Taux de swap USD
    "USD_2Y": "DSWP2",      # US 2-Year Swap Rate
    "USD_5Y": "DSWP5",      # US 5-Year Swap Rate
    "USD_10Y": "DSWP10",    # US 10-Year Swap Rate
    "USD_30Y": "DSWP30",    # US 30-Year Swap Rate
    
    # Taux EUR (EURIBOR et équivalents)
    "EUR_3M": "IR3TIB01EZM156N",   # Euro 3-month
    
    # Taux GBP
    "GBP_3M": "IR3TIB01GBM156N",   # UK 3-month
    
    # Taux JPY
    "JPY_3M": "IR3TIB01JPM156N",   # Japan 3-month
}

print("="*70)
print("TÉLÉCHARGEMENT DES TAUX DE SWAP")
print("="*70)

# Téléchargement des données
data_dict = {}
for nom, ticker in tickers.items():
    try:
        print(f"Téléchargement: {nom} ({ticker})...")
        serie = fred.get_series(ticker)
        if serie is not None and len(serie) > 0:
            data_dict[nom] = serie
        else:
            print(f"  ⚠️  Pas de données pour {nom}")
    except Exception as e:
        print(f"  ❌ Erreur pour {nom}: {e}")

# Créer le DataFrame
data = pd.DataFrame(data_dict)

# Nettoyage et préparation
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Forward fill pour propager les valeurs
data = data.ffill()

# Supprimer les lignes avec toutes les valeurs manquantes
data = data.dropna(how="all")

# Filtrer sur les 2 dernières semaines
date_debut = pd.Timestamp.today() - pd.DateOffset(weeks=2)
date_fin = pd.Timestamp.today()
data_filtered = data.loc[date_debut:date_fin]

# Supprimer les colonnes qui n'ont que des NaN sur la période
data_filtered = data_filtered.dropna(axis=1, how='all')

print(f"\n✅ Données téléchargées: {list(data_filtered.columns)}")

# Création du graphique
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8))

# Palette de couleurs
colors = ['#00ff41', '#ff006e', '#ffbe0b', '#8338ec', '#fb5607', '#3a86ff', '#06ffa5', '#ff1744']

# Tracé des courbes
for i, col in enumerate(data_filtered.columns):
    series_clean = data_filtered[col].dropna()
    if len(series_clean) > 0:
        ax.plot(series_clean.index, series_clean.values, 
                label=col, linewidth=2.5, color=colors[i % len(colors)], alpha=0.9)

# Mise en forme
ax.set_title("Taux de Swap - Principales Devises (2 dernières semaines)", 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Taux (%)", fontsize=13)
ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
ax.grid(True, linestyle='--', alpha=0.4)

# Limites de l'axe x
ax.set_xlim(date_debut, date_fin)

# Améliorer la lisibilité des dates
fig.autofmt_xdate()

plt.tight_layout()
plt.show()

# Afficher quelques statistiques
print("\n" + "="*70)
print("ANALYSE COURT TERME (2 DERNIÈRES SEMAINES)")
print("="*70)

print("\n📊 Valeurs actuelles (dernière observation disponible):")
print("-" * 70)
for col in data_filtered.columns:
    serie = data_filtered[col].dropna()
    if len(serie) > 0:
        derniere_valeur = serie.iloc[-1]
        derniere_date = serie.index[-1]
        print(f"{col:12} : {derniere_valeur:6.2f}%  (au {derniere_date.strftime('%Y-%m-%d')})")

print("\n📈 Variation sur 2 semaines (en points de base):")
print("-" * 70)
for col in data_filtered.columns:
    serie = data_filtered[col].dropna()
    if len(serie) > 1:
        variation = (serie.iloc[-1] - serie.iloc[0]) * 100
        signe = "+" if variation > 0 else ""
        
        # Calculer la variation quotidienne moyenne
        jours_trading = (serie.index[-1] - serie.index[0]).days
        var_quotidienne = variation / jours_trading if jours_trading > 0 else 0
        
        print(f"{col:12} : {signe}{variation:6.1f} bps  (≈ {signe}{var_quotidienne:.1f} bps/jour)")
    else:
        print(f"{col:12} : Données insuffisantes")

# Ajouter volatilité court terme
print("\n📊 Volatilité sur 2 semaines (écart-type en bps):")
print("-" * 70)
for col in data_filtered.columns:
    serie = data_filtered[col].dropna().pct_change() * 10000  # En bps
    if len(serie) > 1:
        volatilite = serie.std()
        print(f"{col:12} : {volatilite:6.2f} bps")

print("\n💡 Note: Analyse court terme sur les 2 dernières semaines.")
print("    Les données internationales étant mensuelles, les variations peuvent")
print("    sembler nulles si aucune nouvelle publication n'est intervenue.")

# Suggestion de sources alternatives
print("\n" + "="*70)
print("SOURCES ALTERNATIVES POUR LES TAUX DE SWAP")
print("="*70)
print("""
