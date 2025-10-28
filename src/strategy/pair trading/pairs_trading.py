import vectorbt as vbt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy.optimize import minimize

plt.style.use('dark_background')

def get_prices(tickers, start='2025-01-01', end='2025-25-10', interval='1h'):
    price_data = {}
    for ticker in tickers:
        data = vbt.YFData.download(ticker, start=start, end=end, interval=interval).get('Close')
        price_data[ticker] = data
    return pd.DataFrame(price_data)

def compute_adf(series):
    result = adfuller(series.dropna())
    return {'adf_stat': result[0], 'p_value': result[1], 'critical_values': result[4]}

def get_regression(serie1, serie2):
    data = pd.concat([serie1, serie2], axis=1).dropna()
    X = sm.add_constant(data.iloc[:,0])
    y = data.iloc[:,1]
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    return model, residuals

def get_zscore(residuals):
    return (residuals - residuals.mean()) / residuals.std()

def llh(params, X, dt):
    theta, mu, sigma = params
    deltaX = np.diff(X)
    X = X[:-1]
    f = -0.5 * np.log(2*np.pi*sigma**2*dt) - ((deltaX - theta*(mu-X)*dt)**2)/(2*sigma**2*dt)
    res = -f.sum()
    if np.isnan(res) or np.isinf(res):
        res = 1
    return res

def method_moment(X, dt):
    deltaX = np.diff(X)
    X = X[:-1]
    mu = X.mean()
    exog = (mu - X) * dt
    model = sm.OLS(deltaX, exog).fit()
    theta = model.params[0]
    resid = deltaX - theta * exog
    sigma = resid.std() / np.sqrt(dt)
    return theta, mu, sigma

# Exécution
tickers = ['PEP','KO']
prices = get_prices(tickers)
serie1 = prices[tickers[0]]
serie2 = prices[tickers[1]]

adf_serie1 = compute_adf(serie1)
adf_serie2 = compute_adf(serie2)

results, residuals = get_regression(serie1, serie2)
alpha = results.params[0]
beta = results.params[1]

adf_residuals = compute_adf(residuals)
z_score = get_zscore(residuals)

theta, mu, sigma = method_moment(z_score, dt=1/252)
res = minimize(llh, method_moment(z_score, dt=1/252), args=(z_score,1/252), method='Nelder-Mead')

half_life = np.log(2)/theta

# # Visualisation résidus et z-score
# plt.figure(figsize=(14,6))
# plt.subplot(2,1,1)
# plt.plot(residuals.index, residuals, label='Residuals')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylabel("Residuals")
# plt.title('Residuals from Regression')
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(z_score.index, z_score, label='Z-Score', color='orange')
# plt.axhline(0, color='black', linestyle='--')
# plt.axhline(1.0, color='red', linestyle='--')
# plt.axhline(-1.0, color='green', linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylabel("Z-Score")
# plt.title('Z-Score of Residuals')
# plt.legend()
# plt.show()

# Signaux adaptatifs
window = 60
rolling_std = z_score.rolling(window=window).std()
entry_threshold = 2 * rolling_std
exit_threshold = (-1.2 * rolling_std)

signals = pd.Series(0, index=z_score.index)
signals[z_score > entry_threshold] = -1
signals[z_score < -entry_threshold] = 1
signals[z_score.abs() < exit_threshold] = 0

positions = signals.replace(to_replace=0, method='ffill').fillna(0)

# Visualisation signaux
plt.figure(figsize=(14,6))
plt.plot(z_score, label='Z-Score', color='orange')
plt.plot(entry_threshold, '--', color='red', label='Upper threshold')
plt.plot(exit_threshold, '--', color='green', label='Lower threshold')
plt.fill_between(z_score.index, -exit_threshold, exit_threshold, color='gray', alpha=0.2)
plt.scatter(z_score.index, np.where(signals==1, z_score, np.nan), color='green', marker='^', s=60, label='Buy Signal')
plt.scatter(z_score.index, np.where(signals==-1, z_score, np.nan), color='red', marker='v', s=60, label='Sell Signal')
plt.legend()
plt.title("Z-Score avec seuils adaptatifs et signaux")
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# Calcul des returns journaliers pour chaque actif
ret1 = serie1.pct_change().fillna(0)
ret2 = serie2.pct_change().fillna(0)

# Positions décalées d'un jour pour simuler exécution
pos1 = -beta * positions.shift(1)
pos2 = positions.shift(1)

# Returns du portefeuille
# Décaler les positions mais aligner avec les prix
positions_shifted = positions.shift(1).reindex(serie2.index).fillna(0)

# Backtest VectorBT
pf = vbt.Portfolio.from_signals(
    close=serie2,
    entries=positions_shifted==1,
    exits=positions_shifted==-1,
    init_cash=1000,
    fees=0.0005,
    freq='D'  # ou '1H' si tes données sont horaires
)
# Affichage des performances
print("Total Return:", pf.total_return())
print("Sharpe Ratio:", pf.sharpe_ratio())
print("Max Drawdown:", pf.max_drawdown())

#equity curve
pf.value().vbt.plot(title="Equity Curve du Spread")