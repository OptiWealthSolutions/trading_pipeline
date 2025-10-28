from fredapi import Fred
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fred = Fred(api_key="e16626c91fa2b1af27704a783939bf72")

taux_config = {
    'USD': {
        'court': 'DGS3MO',     
        'long': 'DGS10',       
        'nom': 'États-Unis'
    },
    'EUR': {
        'court': 'IR3TIB01EZM156N',   
        'long': 'IRLTLT01EZM156N',    
        'nom': 'Zone Euro'
    },
    'GBP': {
        'court': 'IR3TIB01GBM156N',   
        'long': 'IRLTLT01GBM156N',    
        'nom': 'Royaume-Uni'
    },
    'JPY': {
        'court': 'IR3TIB01JPM156N',   
        'long': 'IRLTLT01JPM156N',    
        'nom': 'Japon'
    },
    'CAD': {
        'court': 'IR3TIB01CAM156N',   
        'long': 'IRLTLT01CAM156N',    
        'nom': 'Canada'
    }
}

data_devises = {}

for devise, config in taux_config.items():
    print(f"\nTéléchargement {config['nom']} ({devise})...")
    try:
        court = fred.get_series(config['court'])
        long = fred.get_series(config['long'])
        
        if court is not None and long is not None:
            df = pd.DataFrame({
                'court': court,
                'long': long
            })
            df = df.dropna()
            
            if len(df) > 0:
                data_devises[devise] = df
                print(f"  ✅ {len(df)} observations")
            else:
                print(f"  ⚠️  Pas de données communes")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")

spreads = {}
for devise, df in data_devises.items():
    df['spread'] = df['long'] - df['court']  
    spreads[devise] = df['spread']

df_spreads = pd.DataFrame(spreads)
df_spreads = df_spreads.ffill().dropna(how='all')

date_debut = pd.Timestamp.today() - pd.DateOffset(years=2)
date_fin = pd.Timestamp.today()
df_spreads_recent = df_spreads.loc[date_debut:date_fin]

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(3, 2, 1)
colors_spread = {'USD': '#00ff41', 'EUR': '#ff006e', 'GBP': '#ffbe0b', 
                 'JPY': '#8338ec', 'CAD': '#3a86ff'}

for devise in df_spreads_recent.columns:
    serie = df_spreads_recent[devise].dropna()
    ax1.plot(serie.index, serie.values, label=devise, 
            linewidth=2.5, color=colors_spread.get(devise, '#ffffff'))

ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.set_title('Écarts de Taux (10Y - 3M) - Évolution Temporelle', 
             fontsize=13, fontweight='bold')
ax1.set_ylabel('Spread (%)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 2, 2)
spreads_actuels = {}
for devise in df_spreads_recent.columns:
    serie = df_spreads_recent[devise].dropna()
    if len(serie) > 0:
        spreads_actuels[devise] = serie.iloc[-1]

devises_sorted = sorted(spreads_actuels.items(), key=lambda x: x[1], reverse=True)
devises_names = [d[0] for d in devises_sorted]
spreads_values = [d[1] for d in devises_sorted]
colors_bars = [colors_spread.get(d, '#ffffff') for d in devises_names]

bars = ax2.barh(devises_names, spreads_values, color=colors_bars, alpha=0.8)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.set_title('Spreads Actuels (10Y - 3M)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Spread (%)', fontsize=11)

for i, (bar, val) in enumerate(zip(bars, spreads_values)):
    ax2.text(val, i, f' {val:.2f}%', va='center', fontsize=10)

ax2.grid(True, alpha=0.3, axis='x')

ax3 = plt.subplot(3, 2, 3)
for devise, df in data_devises.items():
    serie = df['court'].loc[date_debut:date_fin].dropna()
    if len(serie) > 0:
        ax3.plot(serie.index, serie.values, label=devise, 
                linewidth=2, color=colors_spread.get(devise, '#ffffff'))

ax3.set_title('Taux Courts (3 Mois)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Taux (%)', fontsize=11)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 2, 4)
for devise, df in data_devises.items():
    serie = df['long'].loc[date_debut:date_fin].dropna()
    if len(serie) > 0:
        ax4.plot(serie.index, serie.values, label=devise, 
                linewidth=2, color=colors_spread.get(devise, '#ffffff'))

ax4.set_title('Taux Longs (10 Ans)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Taux (%)', fontsize=11)
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 2, 5)
for devise in df_spreads_recent.columns:
    serie = df_spreads_recent[devise].dropna()
    if len(serie) > 30:
        volatilite = serie.rolling(window=30).std()
        ax5.plot(volatilite.index, volatilite.values, label=devise, 
                linewidth=2, color=colors_spread.get(devise, '#ffffff'))

ax5.set_title('Volatilité des Spreads (30 jours glissants)', 
             fontsize=13, fontweight='bold')
ax5.set_ylabel('Écart-type (%)', fontsize=11)
ax5.set_xlabel('Date', fontsize=11)
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 2, 6)
corr_matrix = df_spreads_recent.corr()
im = ax6.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

ax6.set_xticks(range(len(corr_matrix.columns)))
ax6.set_yticks(range(len(corr_matrix.columns)))
ax6.set_xticklabels(corr_matrix.columns)
ax6.set_yticklabels(corr_matrix.columns)
ax6.set_title('Corrélation des Spreads entre Devises', 
             fontsize=13, fontweight='bold')

plt.colorbar(im, ax=ax6, label='Corrélation')

plt.tight_layout()
plt.savefig('analyse_ecarts_taux.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: analyse_ecarts_taux.png")
plt.show()

recap = []
for devise, df in data_devises.items():
    df_recent = df.loc[date_debut:date_fin]
    
    if len(df_recent) > 0:
        court_actuel = df_recent['court'].dropna().iloc[-1]
        long_actuel = df_recent['long'].dropna().iloc[-1]
        spread_actuel = long_actuel - court_actuel
        
        df_1m = df_recent.last('30D')
        if len(df_1m) > 1:
            spread_1m_ago = (df_1m['long'].iloc[0] - df_1m['court'].iloc[0])
            var_1m = (spread_actuel - spread_1m_ago) * 100  
        else:
            var_1m = np.nan
        
        spreads_serie = (df_recent['long'] - df_recent['court']).dropna()
        spread_moyen = spreads_serie.mean()
        spread_std = spreads_serie.std()
        
        recap.append({
            'Devise': devise,
            'Nom': taux_config[devise]['nom'],
            'Taux Court': court_actuel,
            'Taux Long': long_actuel,
            'Spread Actuel': spread_actuel,
            'Var 1M (bps)': var_1m,
            'Spread Moyen 2Y': spread_moyen,
            'Volatilité': spread_std
        })

df_recap = pd.DataFrame(recap)
df_recap = df_recap.sort_values('Spread Actuel', ascending=False)

print("\n📊 Situation Actuelle:")
print("-" * 80)
print(df_recap[['Devise', 'Nom', 'Taux Court', 'Taux Long', 'Spread Actuel']].to_string(index=False))

print("\n📈 Variations et Statistiques:")
print("-" * 80)
print(df_recap[['Devise', 'Var 1M (bps)', 'Spread Moyen 2Y', 'Volatilité']].to_string(index=False))

print("\n🎯 OPPORTUNITÉS DE CARRY TRADING:")
print("-" * 80)

devise_max_spread = df_recap.iloc[0]
devise_min_spread = df_recap.iloc[-1]

print(f"\n✅ Spread le plus élevé: {devise_max_spread['Devise']} ({devise_max_spread['Nom']})")
print(f"   → Spread: {devise_max_spread['Spread Actuel']:.2f}%")
print(f"   → Taux 3M: {devise_max_spread['Taux Court']:.2f}% | Taux 10Y: {devise_max_spread['Taux Long']:.2f}%")

print(f"\n⚠️  Spread le plus faible: {devise_min_spread['Devise']} ({devise_min_spread['Nom']})")
print(f"   → Spread: {devise_min_spread['Spread Actuel']:.2f}%")
print(f"   → Taux 3M: {devise_min_spread['Taux Court']:.2f}% | Taux 10Y: {devise_min_spread['Taux Long']:.2f}%")

if devise_min_spread['Spread Actuel'] < 0:
    print(f"\n⚠️  ATTENTION: Courbe inversée pour {devise_min_spread['Devise']}!")
    print("   → Signal potentiel de récession")