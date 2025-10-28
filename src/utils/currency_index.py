from fredapi import Fred
import pandas as pd
import plotly.graph_objects as go


fred = Fred(api_key="e16626c91fa2b1af27704a783939bf72")

# Indices de monnaie disponibles sur FRED
indices = {
    "DXY (Dollar Index)": "DTWEXBGS",      # Broad USD Index
    "EUR Index": "DEXUSEU",                 # USD/EUR (inversé pour index EUR)
    "GBP Index": "DEXUSUK",                 # USD/GBP (inversé pour index GBP)
    "JPY Index": "DEXJPUS",                 # JPY/USD
    "CHF Index": "DEXSZUS",                 # CHF/USD
    "CAD Index": "DEXCAUS",                 # CAD/USD
}

# Période d'analyse
start_date = "2020-01-01"


print("Téléchargement des indices de monnaie...")

data = {}
for nom, ticker in indices.items():
    try:
        serie = fred.get_series(ticker, observation_start=start_date)
        if serie is not None and len(serie) > 0:
            data[nom] = serie
            print(f"✅ {nom}: {len(serie)} observations")
    except Exception as e:
        print(f"❌ {nom}: {e}")

# Créer DataFrame
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

# Ajouter chaque série
for col in df_normalized.columns:
    fig.add_trace(go.Scatter(
        x=df_normalized.index,
        y=df_normalized[col],
        name=col,
        line=dict(width=2.5, color=colors.get(col, "#ffffff")),
        mode='lines',
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Valeur: %{y:.2f}<br>' +
                      '<extra></extra>'
    ))

# Mise en forme
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

# Ajouter ligne de référence à 100
fig.add_hline(
    y=100, 
    line_dash="dash", 
    line_color="red", 
    opacity=0.5,
    annotation_text="Base 100",
    annotation_position="right"
)

# Grille
fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')

# Afficher
fig.show()

# Sauvegarder en HTML
fig.write_html("indices_monnaie.html")
print("\n✅ Graphique interactif sauvegardé: indices_monnaie.html")


print("\n" + "="*70)
print("PERFORMANCE DES DEVISES (depuis le début de période)")
print("="*70)

performance = ((df_normalized.iloc[-1] / 100) - 1) * 100
performance = performance.sort_values(ascending=False)

print("\nPerformance (%):")
print("-" * 70)
for devise, perf in performance.items():
    signe = "+" if perf > 0 else ""
    print(f"{devise:25} : {signe}{perf:6.2f}%")

print("\n📊 Valeurs actuelles (normalisées):")
print("-" * 70)
for devise, valeur in df_normalized.iloc[-1].sort_values(ascending=False).items():
    print(f"{devise:25} : {valeur:7.2f}")