import vectorbt as vbt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TICKERS = ['PEP', 'KO']           # [asset_X, asset_Y] convention used in code (X hedged by Y)
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
INTERVAL = '1d'                   # '1d' or '1h' etc.
LOOKBACK_WINDOW = 252             # window for rolling beta / spread statistics
ENTRY_THRESHOLD_MULT = 2.0        # entry threshold in z-score units
EXIT_THRESHOLD_MULT = 0.5         # exit threshold in z-score units
INIT_CASH = 10000
FEES = 0.001                      # 0.1% fees per trade

# --- UTILITIES ---

def get_prices(tickers, start, end, interval=INTERVAL):
    """Download prices using vectorbt.YFData with minimal output; return Close prices DataFrame."""
    try:
        raw = vbt.YFData.download(tickers, start=start, end=end, interval=interval)
        prices = raw.get('Close')
        if prices is None or prices.empty:
            print("Aucune donnée retournée par YFData.download.")
            return None
        # Remove duplicate columns and drop rows with all NaN
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices = prices.dropna(how='all')
        return prices
    except Exception as e:
        print(f"Erreur téléchargement prix: {e}")
        return None


def compute_adf(series):
    """Run augmented Dickey-Fuller test and return key metrics or None on failure."""
    try:
        s = series.dropna()
        if len(s) < 15:
            return None
        res = adfuller(s)
        return {'adf_stat': res[0], 'p_value': res[1], 'critical_5%': res[4]['5%']}
    except Exception:
        return None


def get_rolling_beta(series_y, series_x, window):
    """
    Rolling beta: Cov(Y,X) / Var(X) using aligned data.
    Returns a Series indexed like the inputs (nan for initial periods).
    """
    df = pd.concat([series_y, series_x], axis=1).dropna()
    rolling_cov = df.iloc[:, 0].rolling(window=window).cov(df.iloc[:, 1])
    rolling_var = df.iloc[:, 1].rolling(window=window).var()
    beta = rolling_cov / rolling_var
    return beta


# --- MAIN STRATEGY LOGIC ---


def pairs_trading(tickers=TICKERS, start=START_DATE, end=END_DATE, interval=INTERVAL,
                  lookback=LOOKBACK_WINDOW,
                  entry_mult=ENTRY_THRESHOLD_MULT, exit_mult=EXIT_THRESHOLD_MULT,
                  init_cash=INIT_CASH, fees=FEES):
    """
    Runs the pairs trading logic and returns (portfolio, signals_df).
    - tickers: [X, Y] ; X is hedged by Y in the formulation used below.
    """
    # 1) Download prices
    prices = get_prices(tickers, start, end, interval)
    if prices is None or prices.empty:
        print("Aucune donnée prix disponible — arrêt.")
        return None, None

    # Ensure both columns exist
    if not all(t in prices.columns for t in tickers):
        print("Certains tickers manquent dans les données téléchargées.")
        return None, None

    # 2) Cointegration indication (ADF) on full series (informational)
    adf_x = compute_adf(prices[tickers[0]])
    adf_y = compute_adf(prices[tickers[1]])
    if adf_x:
        print(f"ADF {tickers[0]} p-value: {adf_x['p_value']:.4f}")
    if adf_y:
        print(f"ADF {tickers[1]} p-value: {adf_y['p_value']:.4f}")

    # 3) Compute rolling beta on log prices (more stable) and build spread
    logp = np.log(prices)
    rolling_beta = get_rolling_beta(logp[tickers[0]], logp[tickers[1]], lookback)
    # align beta with original index by forward filling
    rolling_beta = rolling_beta.reindex(prices.index).fillna(method='ffill').fillna(method='bfill')

    # Spread: using log-prices ensures multiplicative relationships map to additive spread
    spread = logp[tickers[0]] - rolling_beta * logp[tickers[1]]
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()
    z_score = (spread - rolling_mean) / rolling_std

    # drop initial NaNs
    df = pd.concat([spread, rolling_mean, rolling_std, z_score, rolling_beta], axis=1)
    df.columns = ['spread', 'rolling_mean', 'rolling_std', 'z_score', 'rolling_beta']
    df = df.dropna()

    if df.empty or len(df) < lookback // 2:
        print("Pas assez de données après nettoyage pour générer des signaux.")
        return None, None

    # 4) Signals: fixed thresholds in z-score units
    entry_threshold = entry_mult
    exit_threshold = exit_mult

    signals = pd.Series(0, index=df.index)
    signals[df['z_score'] > entry_threshold] = -1  # spread high -> short asset0 (PEP), long asset1 (KO)
    signals[df['z_score'] < -entry_threshold] = 1  # spread low -> long asset0, short asset1
    # explicit exit zone
    signals[(df['z_score'] < exit_threshold) & (df['z_score'] > -exit_threshold)] = 0

    # forward-fill positions; no position before first non-zero signal
    positions = signals.replace(to_replace=0, method='ffill').fillna(0)

    # 5) Backtest: compute portfolio returns by applying positions and hedge ratio (beta)
    # Use percent returns aligned to df.index
    returns = prices.pct_change().reindex(df.index).fillna(0)

    # shift positions to simulate execution at next bar
    positions_shifted = positions.shift(1).reindex(returns.index).fillna(0)

    # align beta to returns index (use rolling_beta from df)
    beta_aligned = df['rolling_beta'].reindex(returns.index).fillna(method='ffill').fillna(1.0)

    # portfolio returns: position on asset0 is -positions*beta, on asset1 is positions
    # (using convention consistent with spread = asset0 - beta*asset1)
    port_rets = (positions_shifted * returns[tickers[1]]) + (-positions_shifted * beta_aligned * returns[tickers[0]])

    equity_curve = (1 + port_rets.fillna(0)).cumprod() * init_cash
    pf = pd.DataFrame({'value': equity_curve})

    # Calculs de performance manuels
    total_return = (pf['value'].iloc[-1] / init_cash - 1) * 100
    sharpe_ratio = (port_rets.mean() / port_rets.std()) * np.sqrt(252) if port_rets.std() != 0 else 0
    max_drawdown = ((pf['value'] / pf['value'].cummax()) - 1).min() * 100

    print("\n--- Performance Summary (Manual Calculation) ---")
    print(f"Total Return [%]: {total_return:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown [%]: {max_drawdown:.2f}")

    # 6) Build signals DataFrame for output
    signals_df = pd.DataFrame({
        'price_x': prices[tickers[0]].reindex(df.index),
        'price_y': prices[tickers[1]].reindex(df.index),
        'spread': df['spread'],
        'z_score': df['z_score'],
        'beta': df['rolling_beta'],
        'raw_signal': signals,
        'position': positions_shifted  # actual position used to compute returns
    }).dropna()

    # 7) Print performance summary
    print("\n--- Performance summary ---")
    stats = pf.stats()
    # Print a filtered subset if available
    keys = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Total # of Trades']
    avail = [k for k in keys if k in stats.index]
    print(stats.loc[avail])

    # 8) Plot: equity + z-score + signals
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    prices[tickers].plot(ax=axs[0], title=f"Prix: {tickers[0]} (X) et {tickers[1]} (Y)")
    axs[0].set_ylabel("Prix")

    axs[1].plot(df.index, df['z_score'], color='cyan', label='Z-score')
    axs[1].axhline(entry_threshold, color='red', linestyle='--', label='Seuil Entrée')
    axs[1].axhline(-entry_threshold, color='red', linestyle='--')
    axs[1].axhline(exit_threshold, color='green', linestyle='--', label='Seuil Sortie')
    axs[1].axhline(-exit_threshold, color='green', linestyle='--')
    # plot buy/sell markers
    buys = signals_df['position'].diff() == 1
    sells = signals_df['position'].diff() == -1
    axs[1].scatter(signals_df.index[buys], signals_df['z_score'][buys], marker='^', color='lime', s=80, label='Achat')
    axs[1].scatter(signals_df.index[sells], signals_df['z_score'][sells], marker='v', color='red', s=80, label='Vente')
    axs[1].legend()
    axs[1].set_ylabel("Z-score")

    pf.value().vbt.plot(ax=axs[2], title='Equity Curve (pairs portfolio)')
    axs[2].set_ylabel("Valeur du portefeuille")

    plt.tight_layout()
    plt.show()

    # 9) Save signals safely
    if not signals_df.empty:
        signals_df.to_csv('pairs_trading_signals_output.csv')
        print("Signaux sauvegardés dans 'pairs_trading_signals_output.csv'")
    else:
        print("Aucun signal généré — fichier non sauvegardé.")

    return pf, signals_df


if __name__ == "__main__":
    # Example run
    pf, signals = pairs_trading()
    if signals is not None:
        print(signals.head())