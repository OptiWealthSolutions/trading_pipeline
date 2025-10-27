#strategy de pairs trading 
import vectorbt as vbt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adatauller
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from scipy.optimize import minimize, differential_evolution
plt.style.use('dark_background')

def getSerie1():
    price_serie_1 = vbt.YFData.download(
    "PEP",
    start='2010-01-01',
    end='2025-01-01',
    interval='1d'
        ).get('Close')
    #nettoyage de données
    return price_serie_1

def getSerie2():
    price_serie_2 = vbt.YFData.download(
    "KO",
    start='2010-01-01',
    end='2025-01-01',
    interval='1d'
        ).get('Close')
    return price_serie_2
    
# Engel-Granger

def computeAdata(data): #we want p-valu <= 0.05 in order to reject H0
    result = adatauller(data.dropna())
    print(f"Adata Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")


def getRegression(serie1,serie2):
    # realignement des index
    data = pd.concat([serie1, serie2], axis=1).dropna()
    X = sm.add_constant(data.iloc[:, 0])
    y = data.iloc[:, 1]
    model = sm.OLS(y, X)
    results = model.fit()
    residuals = results.resid
    print("OLS done")
    # return both the fitted results object and residuals
    return results, residuals

def getZScore(resid):
    mu = resid.mean()
    resi_std = resid.std()
    z_score = (resid - mu) / resi_std
    return z_score


# Chargement des séries avec périodes alignées
serie1 = getSerie1()
serie2 = getSerie2()

# Calcul Adata sur les séries
print("Adata test for serie1:")
computeAdata(serie1)
print("\nAdata test for serie2:")
computeAdata(serie2)

# Calcul des résidus de la regression
results, residual = getRegression(serie1, serie2)
alpha = results.params[0]
beta = results.params[1]

print("\nAdata test for residuals:")
computeAdata(residual)

# Calcul du z-score
z_score = getZScore(residual)

def llh(params,X,dt):
    theta, mu ,sigma = params
    deltaX = np.diff(X)
    X = X[:-1]
    f = -0.5 * np.log(2 * np.pi * sigma**2 * dt) - ((deltaX - theta*(mu - X)*dt)**2) / (2 * sigma**2 * dt)
    res = -f.sum() #scipy ne peut que faire des minimisation donc on passe par l'ivnerse de la fonction '-f'
    if np.isnan(res) or np.isinf(res):
        res = 1
    return res

def method_moment(X,dt):
    deltaX = np.diff(X)
    X = X[:-1]
    mu = X.mean()
    exog = (mu - X) * dt
    model = sm.OLS(endog=deltaX, exog=exog)
    res = model.fit()
    theta = res.params[0]
    resid = deltaX - theta * exog
    sigma = resid.std() / np.sqrt(dt)
    return theta,mu,sigma

print(method_moment(residual,dt=(1/252)))
res = minimize(llh, method_moment(z_score,dt=(1/252)), args=(z_score, 1/252), method='Nelder-Mead')
print("minimization summary : \n \n",res)

#calcul de la demi vie (tempmoye pour que l'ecart se reduise de moitié)
#halg life = ln(2)/theta  = y jours
half_life = np.log(2)/method_moment(z_score,dt=(1/252))[0]
print("\n demi vie (jours) :",half_life)

# Visualisations
plt.figure(figsize=(14,6))

plt.subplot(2,1,1)
plt.plot(residual.index, residual, label='Residuals')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylabel("Residuals")
plt.title('Residuals from Regression')
plt.legend()

plt.subplot(2,1,2)
plt.plot(z_score.index, z_score, label='Z-Score', color='orange')
plt.axhline(0, color='black', linestyle='--')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylabel("Z-Score")
plt.title('Z-Score of Residuals')
plt.legend()


n_sims = 1000
n_days = 1000 #4 years
dt = 1/252
theta, mu, sigma = method_moment(z_score, dt)
S0 = z_score.iloc[-1]
print(theta,sigma,mu)

def simulate_OU(theta, mu, sigma, X0, dt, n_days, n_sims):
    sims = np.zeros((n_days, n_sims))
    sims[0] = X0
    for t in range(1, n_days):
        dW = np.random.normal(0, np.sqrt(dt), n_sims)
        sims[t] = sims[t-1] + theta * (mu - sims[t-1]) * dt + sigma * dW
    return sims

sims = simulate_OU(theta, mu, sigma, S0, dt, n_days, n_sims)
variation_z_score = (S0-sims[-1, :] )/S0
target = sigma #target doit être exprimé en ecart type ?
prob = (variation_z_score >= target).sum() / n_sims
print(f"probabilité de {target}", prob)
plt.figure(figsize=(12,6))
plt.plot(sims, alpha=0.2, color="skyblue")
plt.plot(sims.mean(axis=1), color="red", linewidth=2, label="Mean Path")
plt.title("Simulations du processus Ornstein-Uhlenbeck sur le Z-Score")
plt.xlabel("Jours")
plt.ylabel("Z-Score simulé")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# =============================
# LOGIQUE DE SIGNAUX ADAPTATIFS
# =============================

# Fenêtre de volatilité
window = 60
rolling_std = z_score.rolling(window=window).std()

# Seuils adaptatifs
entry_threshold = 1.5 * rolling_std
exit_threshold = 0.5 * rolling_std

# Création des signaux
signals = pd.Series(0, index=z_score.index)
signals[z_score > entry_threshold] = -1     # Short spread
signals[z_score < -entry_threshold] = 1     # Long spread
signals[(z_score.abs() < exit_threshold)] = 0  # Neutre

# Positions cumulées (on maintient la position tant qu'il n'y a pas de sortie)
positions = signals.replace(to_replace=0, method='ffill').fillna(0)

# Visualisation des signaux sur le z-score
plt.figure(figsize=(14,6))
plt.plot(z_score, label='Z-Score', color='orange')
plt.plot(entry_threshold, '--', color='red', label='Upper threshold')
plt.plot(-entry_threshold, '--', color='green', label='Lower threshold')
plt.fill_between(z_score.index, -exit_threshold, exit_threshold, color='gray', alpha=0.2)
plt.scatter(z_score.index, np.where(signals == 1, z_score, np.nan), color='green', label='Buy Signal', marker='^', s=60)
plt.scatter(z_score.index, np.where(signals == -1, z_score, np.nan), color='red', label='Sell Signal', marker='v', s=60)
plt.legend()
plt.title("Z-Score avec seuils adaptatifs et signaux de trading")
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Évaluation des performances sur le spread (backtest corrigé - positions asset-level)
# On re-synchronise les séries et on calcule les rendements en tenant compte des positions sur chaque actif
data = pd.concat([serie1, serie2], axis=1).dropna()
data.columns = ['serie1', 'serie2']

# Réindex positions/signals sur data
signals = signals.reindex(data.index).fillna(0)
positions = positions.reindex(data.index).fillna(method='ffill').fillna(0)

# Paramètres de la régression
alpha = results.params[0]
beta = results.params[1]

# Holdings (units) : pour une position = 1 on prend long 1 share de serie2 et short beta shares de serie1
# Exécution au jour t+1 -> shift positions
pos_exec = positions.shift(1).fillna(0)
h2 = pos_exec * 1.0            # holdings on serie2 (long = +1)
h1 = pos_exec * (-beta)        # holdings on serie1 (negative = short beta)

# Prix et variations
p1 = data['serie1']
p2 = data['serie2']
dp1 = p1.diff()
dp2 = p2.diff()
p1_prev = p1.shift(1)
p2_prev = p2.shift(1)

# P&L absolu
pnl = h2 * dp2 + h1 * dp1

# Gross notional for normalization (avoid division by zero)
gross_notional = (abs(h2) * p2_prev + abs(h1) * p1_prev).replace(0, np.nan)

# Strategy returns as P&L / gross_notional
returns = pnl / gross_notional
returns = returns.fillna(0)

# Coûts de transaction: fraction du notional au moment du trade
transaction_cost = 0.005  # 5 bps
trade_signal = positions.diff().abs()
trade_cost = trade_signal.shift(1).fillna(0) * transaction_cost
returns = returns - trade_cost

# =============================
# GESTION DU CAPITAL ET RISQUE PAR TRADE
# =============================

# Capital initial
initial_capital = 10_000.0
capital = [initial_capital]

# Paramètre de risque (fraction du capital risquée par trade)
risk_fraction = 0.02  # 2% du capital par trade

# Calcul dynamique du capital
for i in range(1, len(returns)):
    prev_cap = capital[-1]
    # Taille de position proportionnelle au capital
    trade_return = returns.iloc[i] * (risk_fraction * prev_cap)
    new_cap = prev_cap + trade_return
    capital.append(new_cap)

capital = pd.Series(capital, index=returns.index)

# Performance cumulée et capital total
plt.figure(figsize=(12,6))
plt.plot(capital, label='Capital en €')
plt.title("Évolution du capital avec gestion du risque (2% par trade)")
plt.xlabel("Date")
plt.ylabel("Capital (€)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()