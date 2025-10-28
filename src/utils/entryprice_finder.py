import pandas as pd
import numpy as np

def optimal_entry(price, signal_prob, atr, capital, risk_pct=0.0025, k=1.2, direction="long"):
    risk_amount = capital * risk_pct

    if direction == "long":
        stop = price - k * atr
        expected_move = atr * (1.2 * signal_prob - 1)
        entry = price - expected_move
    elif direction == "short":
        stop = price + k * atr
        expected_move = atr * (1.2 * signal_prob - 1)
        entry = price + expected_move
    else:
        raise ValueError("direction must be 'long' or 'short'")

    position_size = risk_amount / abs(price - stop)
    return {"entry": entry, "stop": stop, "size": position_size}

optimal_entry(1.1659, 0.55, 0.0072, 885, direction="short")