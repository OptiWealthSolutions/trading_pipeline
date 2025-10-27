
#package importation
import vectorbt as vbt
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import os 
class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0] #nb de sample grace a la taille de la df_features
        test_size = n_samples // self.n_splits
        embargo = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size
            test_idx = np.arange(test_start, test_end)
            train_idx = np.arange(0, test_start)
            if test_end + embargo < n_samples:
                train_idx = np.concatenate([train_idx, np.arange(test_end + embargo, n_samples)])
            yield train_idx, test_idx
        return

class SampleWeights():
    def __init__(self, labels, features, timestamps):
        # Aligner correctement index et timestamps
        self.timestamps = pd.Series(timestamps, index=timestamps)
        self.labels = pd.Series(labels, index=timestamps)
        self.features = features
        self.n_samples = len(labels)
        self.df = pd.DataFrame(features, index=timestamps)
        self.df['labels'] = self.labels

    def getIndMatrix(self, label_endtimes=None):
        if label_endtimes is None:
            label_endtimes = self.timestamps
        molecules = label_endtimes.index
        all_ranges = [(start, label_endtimes[start]) for start in molecules]

        # Créer un DatetimeIndex unique pour toutes les périodes
        all_times = pd.date_range(self.timestamps.min(), self.timestamps.max(), freq='D')
        indicator = np.zeros((len(molecules), len(all_times)), dtype=np.uint8)
        time_pos = {dt: idx for idx, dt in enumerate(all_times)}

        for sample_idx, (start, end) in enumerate(all_ranges):
            if pd.isna(start) or pd.isna(end):
                continue
            rng = pd.date_range(start, end, freq='D')
            valid_idx = [time_pos[dt] for dt in rng if dt in time_pos]
            if valid_idx:
                indicator[sample_idx, valid_idx] = 1

        # S'assurer qu'aucune ligne n'est vide
        indicator[indicator.sum(axis=1) == 0, 0] = 1
        return pd.DataFrame(indicator, index=molecules, columns=all_times)

    def getAverageUniqueness(self, indicator_matrix):
        timestamp_usage_count = indicator_matrix.sum(axis=0).values
        mask = indicator_matrix.values.astype(bool)
        uniqueness_matrix = np.divide(
            mask, 
            timestamp_usage_count,
            out=np.zeros_like(mask, dtype=float),
            where=timestamp_usage_count > 0
        )
        avg_uniqueness = uniqueness_matrix.sum(axis=1) / (mask.sum(axis=1) + 1e-10)
        return pd.Series(avg_uniqueness, index=indicator_matrix.index)

    def getRarity(self):
        returns = self.df['labels']
        abs_returns = returns.abs()
        if abs_returns.sum() == 0:
            return pd.Series(np.ones(len(returns))/len(returns), index=returns.index)
        return abs_returns / abs_returns.sum()

    def getSequentialBootstrap(self, indicator_matrix, sample_length=None, random_state=42, n_simulations=10000):
        np.random.seed(random_state)
        n_samples = indicator_matrix.shape[0]
        if sample_length is None:
            sample_length = n_samples
        avg_uniqueness = self.getAverageUniqueness(indicator_matrix)
        probabilities = avg_uniqueness / avg_uniqueness.sum()

        all_choices = np.random.choice(
            n_samples,
            size=n_simulations * sample_length,
            replace=True,
            p=probabilities.values
        ).reshape(n_simulations, sample_length)

        counts = np.bincount(all_choices.ravel(), minlength=n_samples)
        sample_weights = pd.Series(counts, index=indicator_matrix.index)
        sample_weights /= sample_weights.sum() if sample_weights.sum() > 0 else 1
        return sample_weights

    def getRecency(self, decay=0.01):
        time_delta = (self.timestamps.max() - self.timestamps).dt.days
        weights = np.exp(-decay * time_delta)
        return pd.Series(weights, index=self.timestamps.index) / weights.sum()



class MomentumStrategy:
    def __init__(self, ticker):
        self.ticker = ticker
        self.PERIOD = "15y"
        self.INTERVAL = "1d"
        self.SHIFT = 4
        self.lags = [12]
        self.df = self.getDataLoad()
        self.df_features = pd.DataFrame()
        self.meta_df = pd.DataFrame()
        self.meta_features_df = pd.DataFrame()
        self.last_proba = None
        self.last_proba_meta = None


    # --- Data Loading ---
    def getDataLoad(self):
        df = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL, progress= False)
        df = df.dropna()
        Q1 = df['Close'].quantile(0.25)
        Q3 = df['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['return'] = df['Close'].pct_change(self.SHIFT).shift(-self.SHIFT)
        df.dropna(inplace=True)
        return df

    # --- Feature Engineering ---
    def getRSI(self):
        self.df['RSI'] = self.df['Close'].diff().pipe(lambda x: x.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean() / self.df['Close'].diff().pipe(lambda x: -x.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        self.df['RSI'] = 100 - (100 / (1 + self.df['RSI']))
        return self.df
    
    def PriceMomentum(self):
        self.df['PriceMomentum'] = (self.df['Close'] / self.df['Close'].shift(12) - 1) * 100
        return self.df
    
    def getLagReturns(self):
        for n in self.lags:
            self.df[f'RETURN_LAG_{n}'] = np.log(self.df['Close'] / self.df['Close'].shift(n))
        return self.df
    
    def PriceAccel(self):
        self.df['velocity'] = self.df['log_return']
        self.df['acceleration'] = self.df['log_return'].diff()    
        return self.df
    
    def getPct52WeekHigh(self):
        w_high = self.df['High'].rolling(window=252).max()
        self.df['Pct52WeekHigh'] = self.df['Close'] / w_high
        return self.df
    
    def getPct52WeekLow(self):
        w_low = self.df['Low'].rolling(window=252).min()
        self.df['Pct52WeekLow'] = self.df['Close'] / w_low
        return self.df
    
    def get12MonthPriceMomentum(self):
        self.df['12MonthPriceMomentum'] = (self.df['Close'] / self.df['Close'].shift(252) - 1) * 100
        return self.df
    
    def getVol(self):
        self.df['MonthlyVol'] = self.df['Close'].pct_change().rolling(window=20).std()
        return self.df
        
    def getMacroData(self):
        import pandas_datareader.data as web

        # Télécharger DXY et VIX via yfinance
        dxy = yf.download("DX-Y.NYB", period=self.PERIOD, interval="1d",progress= False)['Close']

        # Télécharger TWI via FRED
        try:
            twi = web.DataReader("DTWEXBGS", "fred")
            twi = twi.resample("D").last()
            twi = twi['DTWEXBGS']
        except:
            twi = pd.Series(index=self.df.index, data=np.nan)

        # Réindexer et forward-fill
        self.df['DXY'] = dxy.reindex(self.df.index, method='ffill')
        self.df['TWI'] = twi.reindex(self.df.index, method='ffill')
        return self.df

    # --- Dataset Preparation ---
    def getFeaturesDataSet(self):
        self.df_features = self.df.drop(['High', 'Low', 'Open', 'Volume', 'Close', 'Return', 'Velocity'], axis=1, errors='ignore')
        return self.df_features


    # --- Labeling / Weights ---
    def getLabels(self, max_hold_days=12, stop_loss=0.01, profit_target=0.05, volatility_scaling=True):
        # Barrières adaptatives à la volatilité, avec enregistrement du facteur d'ajustement
        prices = self.df['Close']
        n = len(prices)
        prices_array = prices.values
        labels = np.zeros(n)
        entry_dates, exit_dates, entry_prices, exit_prices = [], [], [], []
        returns_pct, hold_days, barrier_hit, vol_adj_arr = [], [], [], []

        def _find_first_barrier_hit(prices, entry_idx, profit_target, stop_loss, max_hold):
            entry_price = prices[entry_idx]
            end_idx = min(entry_idx + max_hold, len(prices) - 1)
            for i in range(entry_idx + 1, end_idx + 1):
                raw_ret = (prices[i] - entry_price) / entry_price
                if raw_ret >= profit_target:
                    return 1, i  # Profit hit
                elif raw_ret <= -stop_loss:
                    return -1, i  # Stop loss hit
            return 0, end_idx  # Time barrier hit

        # Calcul de la volatilité roulante si scaling activé
        if volatility_scaling:
            returns = prices.pct_change()
            vol = returns.rolling(20).std().fillna(returns.std())
            vol_filled = vol.fillna(method='bfill').fillna(method='ffill')
        else:
            vol_filled = None  # Pas utilisé

        for i in range(n):
            # Skip si prix manquant
            if np.isnan(prices_array[i]):
                labels[i] = 0
                entry_dates.append(prices.index[i])
                exit_dates.append(prices.index[i])
                entry_prices.append(prices_array[i])
                exit_prices.append(prices_array[i])
                returns_pct.append(0)
                hold_days.append(0)
                barrier_hit.append('NaN')
                vol_adj_arr.append(1.0)
                continue

            # Ajustement des barrières selon volatilité
            if volatility_scaling:
                vol_value = float(vol_filled.iloc[i])
                vol_adj = max(vol_value / 0.02, 0.5)
                profit_adj = profit_target * vol_adj
                loss_adj = stop_loss * vol_adj
            else:
                profit_adj = profit_target
                loss_adj = stop_loss
                vol_adj = 1.0

            label, exit_idx = _find_first_barrier_hit(
                prices_array, i, profit_adj, loss_adj, max_hold_days
            )

            labels[i] = label
            entry_dates.append(prices.index[i])
            exit_dates.append(prices.index[exit_idx])
            entry_prices.append(prices_array[i])
            exit_prices.append(prices_array[exit_idx])
            raw_return = (prices_array[exit_idx] - prices_array[i]) / prices_array[i]
            returns_pct.append(raw_return)
            hold_days.append(exit_idx - i)
            barrier_hit.append(['Time', 'Profit', 'Loss'][label + 1])
            vol_adj_arr.append(vol_adj)

        # Mise à jour du DataFrame
        self.df['Target'] = labels
        self.df['label_entry_date'] = entry_dates
        self.df['label_exit_date'] = exit_dates
        self.df['label_entry_price'] = entry_prices
        self.df['label_exit_price'] = exit_prices
        self.df['label_return'] = returns_pct
        self.df['label_hold_days'] = hold_days
        self.df['label_barrier_hit'] = barrier_hit
        self.df['vol_adjustment'] = vol_adj_arr

        return self.df
    def getSampleWeight(self, decay=0.01):
        """Compute and store sample weights using the SampleWeights helper class.

        This replaces the previous implementation that incorrectly called methods
        (getIndMatrix, getRarity, ...) on the MomentumStrategy instance.
        """
        # Instantiate helper with labels, features and timestamps
        sw = SampleWeights(labels=self.df['Target'], features=self.df_features, timestamps=self.df.index)

        # Prefer explicit label end times if available (label_exit_date), otherwise leave None
        label_endtimes = None
        if 'label_exit_date' in self.df.columns:
            label_endtimes = self.df['label_exit_date']

        # Build indicator matrix and weight components via the helper
        indicator_matrix = sw.getIndMatrix(label_endtimes=label_endtimes)
        rarity_weights = sw.getRarity()
        recency_weights = sw.getRecency(decay)
        sequential_weights = sw.getSequentialBootstrap(indicator_matrix)

        # Align indices and combine multiplicatively
        common_index = rarity_weights.index.intersection(recency_weights.index).intersection(sequential_weights.index)
        combined = (
            rarity_weights.loc[common_index].fillna(0) *
            recency_weights.loc[common_index].fillna(0) *
            sequential_weights.loc[common_index].fillna(0)
        )

        # Normalize
        if combined.sum() > 0:
            combined = combined / combined.sum()
        else:
            # fallback: uniform weights over full dataset
            combined = pd.Series(np.ones(len(self.df.index)) / len(self.df.index), index=self.df.index)

        # Reindex to the full dataframe index and fill missing with 0
        full_weights = combined.reindex(self.df.index).fillna(0)

        # Store in the main dataframe for later use by PrimaryModel
        self.df['SampleWeight'] = full_weights

        return full_weights
    # --- Primary Model ---
    def PrimaryModel(self, n_splits=5):
        # S'assurer que df_features est bien initialisé
        if not hasattr(self, 'df_features') or self.df_features.empty:
            self.getFeaturesDataSet()
        X = self.df_features.values 
        y = self.df['Target'].values
        self.df['Target'].to_csv(f'target_{self.ticker}.csv')
        sample_weights = self.df['SampleWeight'].values  # poids calculés

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # herite de la classe PurgerKfold pour faire une 
        # cross-validation temporelle avec embargo et purge
        tscv = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
        scores = []
        reports = []
        cms = []
        f1_score_ = []
        last_pred = []
        # gris search CV
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weights[train_idx]
            # model tuning and hyper parameter
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )

            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict(X_test)

            scores.append(accuracy_score(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            cms.append(confusion_matrix(y_test, y_pred))
            f1_score_.append(f1_score(y_test, y_pred, average='weighted'))
        print("\n=== PRIMARY MODEL RESULTS ===")
        print(f"Average Accuracy : {round((np.mean(scores)*100),2)} %")
        # Metrics
        print(f"Average F1 Score : {round(np.mean(f1_score_)*100,2)} %")
        # Générer les signaux du modèle principal pour toutes les dates
        primary_preds = model.predict(X_scaled)
        self.df['primary_signal'] = primary_preds
        last_pred = primary_preds[-1]
        print("Last prediction", last_pred)
        last_proba = model.predict_proba(X_test)[-1]
        print(f"Last prediction probability: {last_proba}")
        self.meta_df = pd.DataFrame(model.predict_proba(X_scaled), index=self.df.index)
        # Stocker les prédictions pour les meta-features
        self.primary_predictions = primary_preds
        self.last_proba = last_proba
        self.last_pred = last_pred
        return self.meta_df, last_proba

    # --- Meta Features ---
    def getEntropy(self):
        probabilities = self.meta_df.values
        # Éviter log(0) en ajoutant un epsilon
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        # Calcul de l'entropie : -sum(p * log(p))
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        self.meta_features_df['prediction_entropy'] = entropy 
        return 

    def getMaxProbability(self):
        max_probs = np.max(self.meta_df.values, axis=1)
        self.meta_features_df['max_probability'] = max_probs 
        return 
    
    def getMarginConfidence(self):
        probs = self.meta_df.values
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # Plus haute - 2ème plus haute
        self.meta_features_df['margin_confidence'] = margin
        return margin
    
    def getF1ScoreDF(self, y_true, y_pred, window_size=50):
        """Calcule le F1-score sur une fenêtre glissante pour tout le dataset"""
        rolling_f1 = []
        for i in range(len(y_pred)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            if end_idx - start_idx >= 10:  # Minimum d'échantillons
                window_f1 = f1_score(
                    y_true[start_idx:end_idx],
                    y_pred[start_idx:end_idx],
                    average='macro'
                )
            else:
                window_f1 = 0.0
            rolling_f1.append(window_f1)
        self.meta_features_df['f1_score'] = rolling_f1
        return rolling_f1

    def getAccuracyDF(self, y_true, y_pred, window_size=50):
        """Calcule l'accuracy sur une fenêtre glissante pour tout le dataset"""
        rolling_acc = []
        for i in range(len(y_pred)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            if end_idx - start_idx >= 10:  # Minimum d'échantillons
                window_acc = accuracy_score(
                    y_true[start_idx:end_idx],
                    y_pred[start_idx:end_idx]
                )
            else:
                window_acc = 0.0
            rolling_acc.append(window_acc)
        self.meta_features_df['accuracy'] = rolling_acc
        return rolling_acc

    def getMetaFeaturesDf(self):
        return self.meta_features_df

    #=== Tableau de comprehension du duo primary et meta model ===
    # 1 et 1 =-> Strong BUY
    # 1 et 0 =-> ignore
    # 0 et 1 =-> no signal
    # 0 et 0 =-> no signal
    # -1 et 1 =-> Strong SELL
    # -1 et 0 =-> ignore

    # --- Meta Labelling ---
    def metaLabeling(self):
        # Utiliser les prédictions du modèle principal pour déterminer les signaux
        model_predictions = self.primary_predictions != 0  # True si le modèle a généré un signal
        actual_profitable = self.df['label_return'] > 0     # True si le trade était profitable
        
        # Meta-label: 1 si signal ET profitable, 0 sinon
        meta_labels = (model_predictions & actual_profitable).astype(int)
        
        # Créer le DataFrame meta_df à partir des dates correspondantes
        self.meta_df = pd.DataFrame(index=self.df.index)
        self.meta_df['meta_label'] = meta_labels
        
        return
    
    # --- Meta Model ---
    def MetaModel(self):
        X = self.meta_features_df.values
        y = self.meta_df['meta_label'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split temporel
        tscv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        scores = []
        reports = []
        cms = []
        f1_score_ = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # model tuning and hyper parameter
            meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
            meta_model.fit(X_train, y_train)
            y_pred = meta_model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            cms.append(confusion_matrix(y_test, y_pred))
            f1_score_.append(f1_score(y_test, y_pred, average='weighted'))
        # Prédictions
        print("\n=== META MODEL RESULTS ===")
        print(f"Average Accuracy Meta Model: {round((np.mean(scores)*100),2)} %")
        print(f"Average F1 Score Meta Model: {round(np.mean(f1_score_)*100,2)} %")
        # Prédire les signaux méta sur tout l'historique
        self.meta_preds = meta_model.predict(X_scaled)
        self.df['meta_signal'] = self.meta_preds
        last_pred = self.meta_preds[-1]
        print(f"Meta Model Last Prediction: {last_pred}")
        last_proba_meta = meta_model.predict_proba(X_test)[-1]
        print(f"Meta Model Last Prediction Probability: {last_proba_meta}")
        # Information Ratio (IR): Measures the consistency of a predictive signal, calculated as the mean of the Information Coefficient divided by its standard deviation (E[IC] / σ[IC])
        #variable for backtesting functionnement
        self.last_proba_meta = last_proba_meta
        self.last_pred_meta = last_pred
        return meta_model, last_proba_meta

    # --- Confidence Score ---
    def computeConfidenceScore(self):
        if self.last_proba is None or self.last_proba_meta is None:
            return None
        primary_conf = float(np.max(self.last_proba)) if hasattr(self.last_proba, '__len__') else float(self.last_proba)
        meta_conf = float(np.max(self.last_proba_meta)) if hasattr(self.last_proba_meta, '__len__') else float(self.last_proba_meta)
        conf_score = primary_conf * meta_conf
        print(f"Confidence Score: {round(conf_score*100,2)} %")
        return conf_score
    
class BetSizing():
    def __init__(self, ticker):
        self.ticker = ticker
        self.capital = None
        self.risk_pct = None
        self.leverage = 30
    # Taille de la position = Taille du compte x Risque du compte / Point d'invalidation
    def getlastPrice(self):
        df = yf.download(self.ticker, period="1d", interval='1m', progress=False)['Close']
        last_price = float(df.iloc[-1])
        return last_price

    def position_size_with_atr(self, capital, risk_pct, entry_price, atr_value, atr_mult=2):
        if atr_value < 0.1:  # heuristic: treat as percentage
            atr_abs = atr_value * entry_price
        else:
            atr_abs = atr_value
        stop_price = entry_price - atr_mult * atr_abs
        risk_amount = capital * risk_pct 
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0, stop_price
        shares = risk_amount // risk_per_share
        max_shares = capital // entry_price
        shares = min(shares, max_shares)
        print(f"Entry Price: {entry_price:.2f}")
        print(f"Stop Loss (ATR {atr_mult}x): {stop_price:.2f}")
        print(f"Capital: {capital} | Risk per trade: {risk_amount:.2f}")
        print(f"Position Size: {int(shares)} shares (max levier {self.leverage}x)")
        return int(shares), stop_price


# === Backtest with vectorbt ===
class backtest():
    def __init__(self, ms):
        self.ms = ms  # stocke l'objet MomentumStrategy
        self.df_backtest = self.ms.df.loc["2012-01-01":"2025-01-01"].copy()
        self.df_backtest['signal'] = 0
        self.df_backtest.loc[(self.df_backtest['primary_signal'] == 1) & (self.df_backtest['meta_signal'] == 1), 'signal'] = 1
        self.df_backtest.loc[(self.df_backtest['primary_signal'] == -1) & (self.df_backtest['meta_signal'] == 1), 'signal'] = -1
        self.entries = self.df_backtest['signal'] == 1
        self.exits   = self.df_backtest['signal'] == -1

    def portfolio(self):
        self.pf = vbt.Portfolio.from_signals(
            close=self.df_backtest['Close'],
            entries=self.entries,
            exits=self.exits,
            init_cash=10_000,
            fees=0.005,
            freq="1D"
        )
        
        print("\n=== VECTORBT BACKTEST RESULTS ===")
        print(self.pf.stats())
        self.df_backtest['portfolio_value'] = self.pf.value()
        sharpe_ratio = self.pf.sharpe_ratio()
        print(f"Sharpe ratio: {sharpe_ratio}")
        self.df_backtest['portfolio_value'].plot()
        plt.show()

        return self.ms


ticker_list = [
     "EURUSD=X", 
        "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
        "NZDUSD=X", "EURJPY=X", "EURGBP=X", "EURCHF=X"
]



def summarize_signal(ms, shares, stop, last_price, capital, risk_pct, conf_score):
    # Get last row index (date)
    primary_signal = ms.df['primary_signal'].iloc[-1] if 'primary_signal' in ms.df.columns else 0
    meta_signal = ms.df['meta_signal'].iloc[-1] if 'meta_signal' in ms.df.columns else 0
    signal = None
    if (primary_signal == 1) and (meta_signal == 1):
        signal = "BUY"
    elif (primary_signal == -1) and (meta_signal == 1):
        signal = "SELL"
    # Only summarize if there is a valid signal
    if signal is not None:
        row = {
            "ticker": ms.ticker,
            "signal": signal,
            "last_price": last_price,
            "shares": shares,
            "stop": stop,
            "confidence_score": conf_score,
            "risk_amount": risk_pct * capital * conf_score
        }
        return pd.DataFrame([row])
    else:
        return pd.DataFrame(columns=["ticker", "signal", "last_price", "shares", "stop", "confidence_score", "risk_amount"])


def main(ticker):
    print(f"\n--- Processing {ticker} ---")
    ms = MomentumStrategy(ticker)
    ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getPct52WeekLow()
    ms.getPct52WeekHigh()
    ms.getVol()
    ms.getMacroData()
    ms.getFeaturesDataSet()
    ms.getLabels()
    ms.getSampleWeight()
    ms.PrimaryModel()
    ms.getEntropy()
    ms.getMaxProbability()
    ms.getMarginConfidence()
    ms.getF1ScoreDF(ms.df['Target'], ms.primary_predictions)
    ms.getAccuracyDF(ms.df['Target'], ms.primary_predictions)
    ms.getMetaFeaturesDf()
    ms.metaLabeling()
    ms.MetaModel()
    conf_score = ms.computeConfidenceScore()

    # BetSizing integration
    bs = BetSizing(ms.ticker)
    last_price = bs.getlastPrice()
    capital = 885
    risk_pct = 0.0025
    ##### ==== #faire un risque adapté au position deja ouverte ====
    if 'log_return' in ms.df.columns:
        atr_value = ms.df['log_return'].rolling(14).std().iloc[-1]
    else:
        atr_value = 0.01
    shares, stop = bs.position_size_with_atr(capital, risk_pct, last_price, atr_value)

    # Summarize signal
    summary_df = summarize_signal(ms, shares, stop, last_price, capital, risk_pct, conf_score)

    # bt = backtest(ms)
    # bt.portfolio()
    return ms, summary_df

if __name__ == "__main__":
    results = {}
    summaries = []
    for ticker in ticker_list:
        try:
            ms, summary_df = main(ticker)
            results[ticker] = ms
            if summary_df is not None and not summary_df.empty:
                summaries.append(summary_df)
        except Exception as e:
            print(f"Erreur sur {ticker}: {e}")

    # Concatenate all summaries and print
    if summaries:
        df_summary = pd.concat(summaries, ignore_index=True)
        print("\n=== SIGNAL SUMMARY ===")
        print(df_summary)
        # Save summary to CSV for each ticker
        for summary in summaries:
            if not summary.empty and "ticker" in summary.columns:
                ticker = summary["ticker"].iloc[0]
                summary.to_csv(f"summary_{ticker}.csv", index=False)
    else:
        print("\nNo valid signals to summarize.")

