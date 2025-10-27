from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pandas as pd

API_KEY = "e16626c91fa2b1af27704a783939bf72"
fred = Fred(api_key=API_KEY)
data = fred.get_series('CPALTT01USM657N')

# Differencing pour stationnarité
data_diff = data.diff().dropna()

# Séparer train/test
train = data_diff[data_diff.index < "2019-01-01"]
test = data_diff[data_diff.index >= "2019-01-01"]

# SARIMAX
order = (2,1,1)
seasonal_order = (1,0,0,12)

model = SARIMAX(train,
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# Prévisions dynamiques
pred = results.get_forecast(steps=len(test))
pred_mean = pred.predicted_mean  # prédictions sur la série différenciée

# Reconstruire les niveaux à partir des prévisions en différences
# point de départ : dernier niveau observé correspondant à la dernière date d'entraînement
last_train_date = train.index[-1]
last_level = data.loc[last_train_date]

# cumsum des différences prédites puis addition du dernier niveau observé
pred_mean_levels = pred_mean.cumsum() + last_level

# Graphique final (niveaux)
plt.figure(figsize=(12,6))
plt.plot(data.index, data, label="CPI : valeurs réelles (niveaux)")
plt.plot(pred_mean_levels.index, pred_mean_levels, label="Prévisions SARIMAX (niveaux)", linestyle="--")

plt.title("CPI : valeurs réelles vs prévisions SARIMAX")
plt.xlabel("Date")
plt.ylabel("CPI")
plt.legend()
plt.show()

# Afficher dernière prévision (niveau)
print("Dernière prévision (niveau) :", pred_mean_levels.iloc[-1])

# MSE calculé sur les niveaux réels vs prévisions en niveaux
test_levels = data.loc[test.index]
mse = mean_squared_error(test_levels, pred_mean_levels)
print(f"Mean Squared Error (MSE) en niveaux : {mse}")