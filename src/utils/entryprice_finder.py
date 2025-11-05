import pandas as pd
import numpy as np

def optimal_entry(price, signal_prob, atr, capital, risk_pct=0.02, k=1.2, direction="long"):
    risk_amount = capital * risk_pct

    atr_pct = atr / price
    stop_distance = max(k * atr, price * 0.01)

    if direction == "long":
        stop = price - stop_distance
        expected_move = atr * (1.2 * signal_prob - 1)
        entry = price - expected_move
    elif direction == "short":
        stop = price + stop_distance
        expected_move = atr * (1.2 * signal_prob - 1)
        entry = price + expected_move
    else:
        raise ValueError("direction must be 'long' or 'short'")

    position_size = risk_amount / abs(price - stop)
    return {"entry": entry, "stop": stop, "size": position_size}

optimal_entry(153.5319, 0.911772, 0.004275, 895, direction="short")
 #
 # USDJPY=X   SELL  153.531998       5  152.219452          0.911772