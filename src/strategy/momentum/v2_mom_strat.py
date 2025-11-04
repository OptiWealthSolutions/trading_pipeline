import vectorbt as vbt
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import warnings
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# --- Configuration Centralisée ---
# Toutes les variables modifiables sont ici pour une meilleure maintenabilité.
CONFIG = {
    "tickers": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"
    ],
    "data": {
        "period": "25y",
        "interval": "1d",
    },
    "features": {
        "rsi_period": 14,
        "momentum_lookback": 12,
        "pct_52w_lookback": 252,
        "vol_window": 20,
        "lags": [12],
    },
    "labeling": {
        "shift": 4,  # Horizon de prédiction
        "max_hold_days": 12,
        "stop_loss_pct": 0.01,
        "profit_target_pct": 0.05,
        "volatility_scaling": True,
    },
    "models": {
        "n_splits": 5,
        "embargo_pct": 0.01,
        "primary": {
            "n_estimators": 100,
            "max_depth": 10,
        },
        "meta": {
            "n_estimators": 100,
            "max_depth": 10,
        }
    },
    "backtest": {
        "start_date": "2012-01-01",
        "end_date": "2025-01-01",
        "init_cash": 10_000,
        "fees": 0.005,
    },
    "execution": {
        "capital": 885,
        "risk_pct": 0.0025,
        "leverage": 30,
        "atr_mult": 2.0,
        "atr_period": 14,
    }
}


# 

class PurgedKFold:
    """Cross-validation temporelle avec purge et embargo pour éviter le data leakage."""
    # (Code inchangé, il est déjà bien conçu)
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        k_fold_size = n_samples // self.n_splits
        embargo = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * k_fold_size
            test_end = test_start + k_fold_size
            test_indices = np.arange(test_start, test_end)
            
            # Purge and embargo
            train_indices = np.arange(0, test_start)
            if test_end + embargo < n_samples:
                train_indices = np.concatenate([train_indices, np.arange(test_end + embargo, n_samples)])
            
            yield train_indices, test_indices

class SampleWeights:
    """Calcule les poids d'échantillons pour le training (unicité, rareté, etc.)."""
    # (Code inchangé, il est déjà bien conçu)
    def __init__(self, labels, features, timestamps):
        self.timestamps = pd.Series(timestamps, index=timestamps)
        self.labels = pd.Series(labels, index=timestamps)
        self.features = features
        self.data = pd.DataFrame(features, index=timestamps)
        self.data['labels'] = self.labels

    def getIndMatrix(self, label_endtimes=None):
        if label_endtimes is None:
            label_endtimes = self.timestamps
        molecules = label_endtimes.index
        # ... (rest of the method is unchanged)
        # Note: Using a daily frequency might be too fine-grained for daily data.
        # A better approach might be to use the index itself if it's a DatetimeIndex.
        # For simplicity, we keep the original logic.
        all_times = pd.date_range(self.timestamps.min(), self.timestamps.max(), freq='D')
        indicator = np.zeros((len(molecules), len(all_times)), dtype=np.uint8)
        time_pos = {dt: idx for idx, dt in enumerate(all_times)}

        for sample_idx, (start, end) in enumerate(zip(molecules, label_endtimes)):
            if pd.isna(start) or pd.isna(end): continue
            rng = pd.date_range(start, end, freq='D')
            valid_idx = [time_pos[dt] for dt in rng if dt in time_pos]
            if valid_idx:
                indicator[sample_idx, valid_idx] = 1
        
        indicator[indicator.sum(axis=1) == 0, 0] = 1
        return pd.DataFrame(indicator, index=molecules, columns=all_times)

    def getAverageUniqueness(self, indicator_matrix):
        timestamp_usage_count = indicator_matrix.sum(axis=0).values
        mask = indicator_matrix.values.astype(bool)
        uniqueness_matrix = np.divide(mask, timestamp_usage_count, out=np.zeros_like(mask, dtype=float), where=timestamp_usage_count>0)
        avg_uniqueness = uniqueness_matrix.sum(axis=1) / (mask.sum(axis=1) + 1e-10)
        return pd.Series(avg_uniqueness, index=indicator_matrix.index)

    def getRarity(self):
        returns = self.data['labels']
        abs_returns = returns.abs()
        if abs_returns.sum() == 0:
            return pd.Series(np.ones(len(returns))/len(returns), index=returns.index)
        return abs_returns / abs_returns.sum()

    def getSequentialBootstrap(self, indicator_matrix, sample_length=None, random_state=42, n_simulations=10000):
        np.random.seed(random_state)
        n_samples = indicator_matrix.shape[0]
        if sample_length is None: sample_length = n_samples
        avg_uniqueness = self.getAverageUniqueness(indicator_matrix)
        probabilities = avg_uniqueness / avg_uniqueness.sum()
        
        all_choices = np.random.choice(n_samples, size=n_simulations * sample_length, replace=True, p=probabilities.values).reshape(n_simulations, sample_length)
        counts = np.bincount(all_choices.ravel(), minlength=n_samples)
        sample_weights = pd.Series(counts, index=indicator_matrix.index)
        sample_weights /= sample_weights.sum() if sample_weights.sum() > 0 else 1
        return sample_weights

    def getRecency(self, decay=0.01):
        time_delta = (self.timestamps.max() - self.timestamps).dt.days
        weights = np.exp(-decay * time_delta)
        return pd.Series(weights, index=self.timestamps.index) / weights.sum()


class TradingStrategyPipeline:
    """
    Pipeline complet pour une stratégie de trading quantitative.
    Orchestre le chargement des données, l'ingénierie de features,
    la création de labels, l'entraînement des modèles (primaire et méta),
    et la génération de signaux.
    """
    def __init__(self, ticker: str, config: dict):
        self.ticker = ticker
        self.config = config
        self.data = pd.DataFrame()
        self.features = pd.DataFrame()
        self.meta_features = pd.DataFrame()
        self.primary_model = None
        self.meta_model = None
        self.last_signal = None
        self.last_confidence = None

    def run(self):
        """Exécute le pipeline complet."""
        print(f"\n--- Lancement du pipeline pour {self.ticker} ---")
        self._load_data()
        self._feature_engineering()
        self._create_labels()
        self._calculate_sample_weights()
        self._train_primary_model()
        self._create_meta_features()
        self._train_meta_model()
        self._generate_final_signal()
        print("--- Pipeline terminé ---")
        return self



    # --- Étape 2: Ingénierie des Features ---
    def _feature_engineering(self):
        """Génère toutes les features techniques."""
        print("Ingénierie des features...")
        cfg_feat = self.config["features"]
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/cfg_feat["rsi_period"], adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/cfg_feat["rsi_period"], adjust=False).mean()
        rs = gain / loss
        self.features['RSI'] = 100 - (100 / (1 + rs))

        # Momentum
        self.features['PriceMomentum'] = (self.data['Close'] / self.data['Close'].shift(cfg_feat["momentum_lookback"]) - 1) * 100
        self.features['12MonthPriceMomentum'] = (self.data['Close'] / self.data['Close'].shift(cfg_feat["pct_52w_lookback"]) - 1) * 100
        
        # Rendements décalés (Lags)
        for n in cfg_feat["lags"]:
            self.features[f'RETURN_LAG_{n}'] = np.log(self.data['Close'] / self.data['Close'].shift(n))

        # Accélération
        self.features['velocity'] = self.data['log_return']
        self.features['acceleration'] = self.data['log_return'].diff()

        # Position par rapport aux extrêmes sur 52 semaines
        w_high = self.data['High'].rolling(window=cfg_feat["pct_52w_lookback"]).max()
        w_low = self.data['Low'].rolling(window=cfg_feat["pct_52w_lookback"]).min()
        self.features['Pct52WeekHigh'] = self.data['Close'] / w_high
        self.features['Pct52WeekLow'] = self.data['Close'] / w_low

        # Volatilité
        self.features['MonthlyVol'] = self.data['Close'].pct_change().rolling(window=cfg_feat["vol_window"]).std()
        
        # Données macro (DXY)
        try:
            dxy = yf.download("DX-Y.NYB", period=self.config["data"]["period"], interval="1d", progress=False)['Close']
            self.features['DXY'] = dxy.reindex(self.data.index, method='ffill')
        except Exception as e:
            print(f"Impossible de charger les données DXY: {e}")
            self.features['DXY'] = np.nan

        self.features = self.features.dropna()
        print(f"{len(self.features.columns)} features créées.")
        return self.features

    def _load_data(self):
        """Télécharge et nettoie les données brutes."""
        print("Chargement des données...")
        data = yf.download(self.ticker, period=self.config["data"]["period"], interval=self.config["data"]["interval"], progress=False)
        
        # Nettoyage des outliers (méthode IQR)
        Q1 = data['Close'].quantile(0.25)
        Q3 = data['Close'].quantile(0.75) # <<< CORRECTION : 0.75 au lieu de 0.25
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)]
        
        # Features de base
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['future_return'] = data['Close'].pct_change(self.config["labeling"]["shift"]).shift(-self.config["labeling"]["shift"])
        
        self.data = data.dropna()
        print(f"Données chargées: {len(self.data)} observations.")
        return self.data

    # --- Étape 3: Création des Labels (Triple Barrier) ---
    def _create_labels(self):
        """Crée les labels en utilisant la méthode des barrières triple."""
        print("Création des labels...")
        cfg_label = self.config["labeling"]
        
        # <<< CORRECTION PRINCIPALE >>>
        # On utilise .values pour obtenir un tableau NumPy. C'est plus sûr et plus rapide
        # pour une itération par position, car cela évite les problèmes d'index de Pandas.
        prices_array = self.data['Close'].values
        # On garde la Series pour les opérations qui ont besoin de l'index (ex: rolling)
        prices_series = self.data['Close']

        n = len(prices_array)
        labels = np.zeros(n)
        
        # Calcul de la volatilité pour l'ajustement des barrières
        if cfg_label["volatility_scaling"]:
            vol = prices_series.pct_change().rolling(self.config["execution"]["atr_period"]).std()
            # <<< CORRECTION : Syntaxe moderne pour fillna >>>
            vol = vol.ffill().bfill()

        for i in range(n):
            # On utilise maintenant le tableau NumPy, garanti d'être un scalaire
            entry_price = prices_array[i] 
            if np.isnan(entry_price): continue
            
            # Ajustement des barrières
            if cfg_label["volatility_scaling"]:
                # On récupère la valeur de la volatilité à la date correspondante
                timestamp_at_i = prices_series.index[i]
                vol_value = vol.loc[timestamp_at_i]
                
                if np.isnan(vol_value):
                    vol_adj = 1.0
                else:
                    vol_adj = max(vol_value / 0.02, 0.5)
                
                profit_target = cfg_label["profit_target_pct"] * vol_adj
                stop_loss = cfg_label["stop_loss_pct"] * vol_adj
            else:
                profit_target = cfg_label["profit_target_pct"]
                stop_loss = cfg_label["stop_loss_pct"]

            # Trouver la première barrière touchée
            for j in range(i + 1, min(i + cfg_label["max_hold_days"], n)):
                current_price = prices_array[j]
                return_pct = (current_price - entry_price) / entry_price
                
                if return_pct >= profit_target:
                    labels[i] = 1  # Profit
                    break
                elif return_pct <= -stop_loss:
                    labels[i] = -1 # Stop Loss
                    break
            else:
                labels[i] = 0 # Time barrier

        self.data['Target'] = labels
        print("Labels créés.")
        return self.data['Target']

# ... (le reste du code reste inchangé) ...

    # --- Étape 4: Calcul des Poids d'Échantillons ---
    def _calculate_sample_weights(self):
        """Calcule et combine les poids d'échantillons."""
        print("Calcul des poids d'échantillons...")
        sw = SampleWeights(
            labels=self.data['Target'],
            features=self.features.values,
            timestamps=self.data.index
        )
        
        indicator_matrix = sw.getIndMatrix()
        rarity_weights = sw.getRarity()
        recency_weights = sw.getRecency()
        sequential_weights = sw.getSequentialBootstrap(indicator_matrix)

        combined_weights = (rarity_weights * recency_weights * sequential_weights).fillna(0)
        if combined_weights.sum() > 0:
            combined_weights /= combined_weights.sum()
        
        self.data['SampleWeight'] = combined_weights.reindex(self.data.index).fillna(0)
        print("Poids d'échantillons calculés.")
        return self.data['SampleWeight']

    # --- Étape 5: Entraînement du Modèle Primaire ---
    def _train_primary_model(self):
        """Entraîne le modèle primaire pour prédire la direction du prix."""
        print("Entraînement du modèle primaire...")
        X = self.features.values
        y = self.data['Target'].values
        sample_weights = self.data['SampleWeight'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tscv = PurgedKFold(n_splits=self.config["models"]["n_splits"], embargo_pct=self.config["models"]["embargo_pct"])
        
        # Note: On entraîne sur le dernier split pour simuler une utilisation en production
        # Une meilleure approche serait d'entraîner sur toutes les données sauf la dernière période de test.
        for train_idx, test_idx in tscv.split(X_scaled):
            pass # On ne fait que récupérer le dernier split
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = sample_weights[train_idx]

        self.primary_model = RandomForestClassifier(
            n_estimators=self.config["models"]["primary"]["n_estimators"],
            max_depth=self.config["models"]["primary"]["max_depth"],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.primary_model.fit(X_train, y_train, sample_weight=w_train)
        
        # Stocker les probabilités pour les méta-features
        primary_proba = self.primary_model.predict_proba(X_scaled)
        self.data['primary_proba_0'] = primary_proba[:, 0]
        self.data['primary_proba_1'] = primary_proba[:, 1]
        self.data['primary_proba_-1'] = primary_proba[:, 2]
        self.data['primary_signal'] = self.primary_model.predict(X_scaled)
        
        print("Modèle primaire entraîné.")
        return self.primary_model

    # --- Étape 6: Création des Méta-Features ---
    def _create_meta_features(self):
        """Génère les features pour le méta-modèle basées sur les prédictions du modèle primaire."""
        print("Création des méta-features...")
        probas = self.data[['primary_proba_-1', 'primary_proba_0', 'primary_proba_1']].values
        
        # Entropie
        epsilon = 1e-10
        probas_clipped = np.clip(probas, epsilon, 1 - epsilon)
        self.meta_features['prediction_entropy'] = -np.sum(probas_clipped * np.log(probas_clipped), axis=1)
        
        # Probabilité maximale
        self.meta_features['max_probability'] = np.max(probas, axis=1)
        
        # Marge de confiance
        sorted_probs = np.sort(probas, axis=1)
        self.meta_features['margin_confidence'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # Performance glissante (F1 et Accuracy)
        window_size = 50
        primary_preds = self.data['primary_signal'].values
        true_labels = self.data['Target'].values
        
        rolling_f1 = []
        rolling_acc = []
        for i in range(len(primary_preds)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            if end_idx - start_idx >= 10:
                window_f1 = f1_score(true_labels[start_idx:end_idx], primary_preds[start_idx:end_idx], average='macro')
                window_acc = accuracy_score(true_labels[start_idx:end_idx], primary_preds[start_idx:end_idx])
            else:
                window_f1, window_acc = 0.0, 0.0
            rolling_f1.append(window_f1)
            rolling_acc.append(window_acc)
            
        self.meta_features['rolling_f1'] = rolling_f1
        self.meta_features['rolling_acc'] = rolling_acc
        
        self.meta_features = self.meta_features.dropna()
        print(f"{len(self.meta_features.columns)} méta-features créées.")
        return self.meta_features

    # --- Étape 7: Entraînement du Méta-Modèle ---
    def _train_meta_model(self):
        """Entraîne le méta-modèle pour filtrer les signaux du modèle primaire."""
        print("Entraînement du méta-modèle...")
        # Le méta-label est 1 si le signal primaire était correct (rentable), 0 sinon.
        self.data['meta_label'] = (self.data['primary_signal'] != 0) & (self.data['future_return'] > 0)
        
        # Aligner les données
        common_index = self.data.index.intersection(self.meta_features.index)
        y = self.data.loc[common_index, 'meta_label'].astype(int).values
        X = self.meta_features.loc[common_index].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tscv = PurgedKFold(n_splits=self.config["models"]["n_splits"], embargo_pct=self.config["models"]["embargo_pct"])
        for train_idx, test_idx in tscv.split(X_scaled):
            pass # Récupérer le dernier split
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        self.meta_model = XGBClassifier(
            n_estimators=self.config["models"]["meta"]["n_estimators"],
            max_depth=self.config["models"]["meta"]["max_depth"],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.meta_model.fit(X_train, y_train)
        
        # Prédire les signaux méta sur tout l'historique
        self.data['meta_signal'] = self.meta_model.predict(X_scaled)
        
        print("Méta-modèle entraîné.")
        return self.meta_model

    # --- Étape 8: Génération du Signal Final et de la Confiance ---
    def _generate_final_signal(self):
        """Combine les signaux primaire et méta pour générer le signal final."""
        print("Génération du signal final...")
        primary = self.data['primary_signal']
        meta = self.data['meta_signal']
        
        # Règle de combinaison
        conditions = [
            (primary == 1) & (meta == 1),
            (primary == -1) & (meta == 1)
        ]
        choices = [1, -1]
        self.data['final_signal'] = np.select(conditions, choices, default=0)
        
        # Calcul du score de confiance pour le dernier signal
        last_row = self.data.iloc[-1]
        if last_row['final_signal'] != 0:
            primary_conf = last_row[['primary_proba_-1', 'primary_proba_0', 'primary_proba_1']].max()
            meta_conf = self.meta_model.predict_proba(self.meta_features.iloc[-1].values.reshape(1, -1))[0].max()
            self.last_confidence = primary_conf * meta_conf
        else:
            self.last_confidence = 0.0
            
        self.last_signal = {
            "ticker": self.ticker,
            "signal": 'BUY' if last_row['final_signal'] == 1 else 'SELL',
            "price": last_row['Close'],
            "confidence": self.last_confidence
        }
        print(f"Signal final généré: {self.last_signal}")
        return self.last_signal

# ==============================================================================
# 4. CLASSES DE BACKTEST ET D'EXÉCUTION
# ==============================================================================

class Backtester:
    """Effectue un backtest de la stratégie avec VectorBT."""
    def __init__(self, pipeline: TradingStrategyPipeline):
        self.pipeline = pipeline
        self.config = pipeline.config
        self.data = self.pipeline.data.loc[self.config["backtest"]["start_date"]:self.config["backtest"]["end_date"]].copy()
        self.portfolio = None

    def run(self):
        """Exécute le backtest et affiche les statistiques."""
        print("\n=== LANCEMENT DU BACKTEST ===")
        entries = self.data['final_signal'] == 1
        exits = self.data['final_signal'] == -1
        
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.data['Close'],
            entries=entries,
            exits=exits,
            init_cash=self.config["backtest"]["init_cash"],
            fees=self.config["backtest"]["fees"],
            freq="1D"
        )
        
        print(self.portfolio.stats())
        self.portfolio.value().plot(title=f"Backtest Performance for {self.pipeline.ticker}")
        plt.show()
        return self.portfolio

class PositionSizer:
    """Calcule la taille de la position basée sur le risque et l'ATR."""
    def __init__(self, config: dict):
        self.config = config

    def calculate_size(self, last_price: float, atr_value: float):
        """Calcule le nombre d'actions à trader."""
        capital = self.config["execution"]["capital"]
        risk_pct = self.config["execution"]["risk_pct"]
        atr_mult = self.config["execution"]["atr_mult"]
        
        risk_amount = capital * risk_pct
        stop_distance = atr_value * atr_mult
        
        if stop_distance == 0: return 0, last_price
        
        position_size = risk_amount / stop_distance
        # Ajustement pour le levier max
        max_shares = (capital * self.config["execution"]["leverage"]) / last_price
        shares = int(min(position_size, max_shares))
        
        stop_price = last_price - stop_distance if shares > 0 else last_price
        
        print(f"Position Sizing: Capital={capital}, Risk={risk_amount:.2f}, StopDist={stop_distance:.4f}")
        print(f"  -> Taille calculée: {shares} actions @ {last_price:.2f}, Stop-Loss @ {stop_price:.2f}")
        
        return shares, stop_price

# ==============================================================================
# 5. FONCTION PRINCIPALE ET LOGIQUE D'EXÉCUTION
# ==============================================================================

def generate_summary_report(signals: list):
    """Génère un rapport PDF et CSV à partir des signaux."""
    if not signals:
        print("\nAucun signal valide à rapporter.")
        return

    summary_df = pd.DataFrame(signals)
    print("\n=== RÉSUMÉ DES SIGNAUX ===")
    print(summary_df)

    # Export CSV
    csv_path = "trading_signals_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nRapport CSV sauvegardé : {csv_path}")

    # Export PDF
    try:
        pdf_path = "trading_signals_summary.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Rapport de Signaux de Trading", styles["Heading1"]))
        elements.append(Spacer(1, 12))

        table_data = [summary_df.columns.to_list()] + summary_df.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        doc.build(elements)
        print(f"Rapport PDF sauvegardé : {pdf_path}")
    except Exception as e:
        print(f"Erreur lors de la génération du PDF : {e}")


def main():
    """Fonction principale d'exécution."""
    final_signals = []
    
    for ticker in CONFIG["tickers"]:
        try:
            # 1. Exécuter le pipeline de trading
            pipeline = TradingStrategyPipeline(ticker, CONFIG).run()
            
            # 2. Lancer le backtest (optionnel, peut être long)
            # backtester = Backtester(pipeline)
            # backtester.run()

            # 3. Si un signal final est généré, calculer la taille de position
            if pipeline.last_signal and pipeline.last_signal['confidence'] > 0.5: # Seuil de confiance
                last_price = pipeline.last_signal['price']
                atr = pipeline.data['log_return'].rolling(CONFIG["execution"]["atr_period"]).std().iloc[-1]
                
                sizer = PositionSizer(CONFIG)
                shares, stop_price = sizer.calculate_size(last_price, atr)
                
                if shares > 0:
                    # Ajouter les infos de taille au signal
                    pipeline.last_signal.update({
                        "shares": shares,
                        "stop_loss": stop_price
                    })
                    final_signals.append(pipeline.last_signal)

        except Exception as e:
            print(f"Une erreur est survenue pour {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Générer le rapport final
    generate_summary_report(final_signals)


if __name__ == "__main__":
    main()
