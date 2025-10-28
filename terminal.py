# trading_pipeline/terminal.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Trading Terminal", initial_sidebar_state="expanded")

# Try to import your existing modules. If absent, provide fallbacks.
try:
    from trading_pipeline.utils import seasonality as seasonality_module
except Exception:
    seasonality_module = None

try:
    from trading_pipeline.utils import volatility as volatility_module
except Exception:
    volatility_module = None

try:
    from trading_pipeline.utils import bonds_rate as bonds_module
except Exception:
    bonds_module = None

# -------------------------
# Helper utilities
# -------------------------
def fetch_price(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df

def fetch_fred(series_code, start=None, end=None):
    if start is None:
        start = datetime.now() - pd.DateOffset(years=2)
    if end is None:
        end = datetime.now()
    try:
        return web.DataReader(series_code, "fred", start, end)
    except Exception as e:
        st.error(f"FRED download error for {series_code}: {e}")
        return pd.DataFrame()

def normalize_series(df):
    return df / df.iloc[0] * 100

def plot_candles(df, title="Price"):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="candles"
    )])
    fig.update_layout(title=title, xaxis_rangeslider_visible=True, template="plotly_dark")
    return fig

def line_plot(df, title, y_label="Value"):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, yaxis_title=y_label, template="plotly_dark")
    return fig

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Terminal Controls")
default_ticker = "EURUSD=X"
ticker = st.sidebar.text_input("Ticker (yfinance)", value=default_ticker)
period = st.sidebar.selectbox("Period", options=["1mo", "3mo", "6mo", "1y", "5y", "10y"], index=3)
interval = st.sidebar.selectbox("Interval", options=["1d", "1h", "30m", "15m"], index=0)
capital = st.sidebar.number_input("Capital (for sizing examples)", value=10000.0)

# FRED controls
st.sidebar.markdown("---")
st.sidebar.header("FRED Macro")
fred_series = st.sidebar.text_input("FRED series (comma separated)", value="DTWEXBGS,DTWEXBPA,DTWEXBGL")
fred_start = st.sidebar.date_input("FRED start", value=(datetime.now() - pd.DateOffset(years=2)).date())
fred_end = st.sidebar.date_input("FRED end", value=datetime.now().date())

# TradingView widget settings
st.sidebar.markdown("---")
st.sidebar.header("TradingView")
tv_symbol = st.sidebar.text_input("TradingView symbol", value="FX:EURUSD")

# -------------------------
# Main UI - Tabs
# -------------------------
tabs = st.tabs(["Dashboard", "Chart", "Volatility", "Seasonality", "Macro (FRED)", "Bonds", "TradingView"])

# -------------------------
# Dashboard tab
# -------------------------
with tabs[0]:
    st.header("Global Dashboard")
    col1, col2, col3 = st.columns((1, 1, 1))
    price_df = fetch_price(ticker, period=period, interval=interval)
    if price_df is None or price_df.empty:
        st.warning("No price data. Check ticker or network.")
    else:
        last = price_df["Close"].iloc[-1]
        ret_1d = price_df["Close"].pct_change().iloc[-1] * 100
        vol_30 = price_df["Close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100
        col1.metric("Last Price", f"{last:.5f}")
        col2.metric("1d Return (%)", f"{ret_1d:.2f}")
        col3.metric("30d Annualized Vol (%)", f"{vol_30:.2f}")

    st.markdown("### Comparative panel")
    # Example list of assets for comparison (can be customized)
    compare_list = st.multiselect("Compare assets", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AAPL", "MSFT"], default=["EURUSD=X", "GBPUSD=X"])
    if compare_list:
        comp_data = {}
        for t in compare_list:
            df = fetch_price(t, period="1y")
            if df is not None and not df.empty:
                comp_data[t] = df["Close"]
        if comp_data:
            comp_df = pd.DataFrame(comp_data).dropna()
            comp_norm = comp_df / comp_df.iloc[0] * 100
            st.plotly_chart(line_plot(comp_norm, "Normalized price comparison (100 start)"), use_container_width=True)

# -------------------------
# Chart tab
# -------------------------
with tabs[1]:
    st.header("Chart")
    if price_df is None or price_df.empty:
        st.warning("No price data to show.")
    else:
        st.plotly_chart(plot_candles(price_df, title=f"{ticker} - {period} {interval}"), use_container_width=True)
        # Add indicators if volatility module exists
        with st.expander("Indicators"):
            ma_period = st.number_input("MA period", value=20, min_value=1)
            price_df["MA"] = price_df["Close"].rolling(ma_period).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df["MA"], name=f"MA{ma_period}"))
            fig.update_layout(template="plotly_dark", title=f"{ticker} Close + MA")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Volatility tab
# -------------------------
with tabs[2]:
    st.header("Volatility")
    if volatility_module is not None:
        try:
            vol_df = volatility_module.compute_volatility(ticker, period=period, interval=interval)
            st.plotly_chart(line_plot(vol_df, f"{ticker} Volatility"), use_container_width=True)
        except Exception as e:
            st.error(f"Volatility module error: {e}")
    else:
        st.info("No volatility module found. Showing realized vol example.")
        if price_df is not None and not price_df.empty:
            rv = price_df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
            st.plotly_chart(line_plot(pd.DataFrame({"RealizedVol": rv}), f"{ticker} Realized Vol"), use_container_width=True)

# -------------------------
# Seasonality tab
# -------------------------
with tabs[3]:
    st.header("Seasonality")
    if seasonality_module is not None:
        try:
            season_df = seasonality_module.compute_seasonality(ticker)
            # Expect season_df indexed by month name
            st.plotly_chart(line_plot(season_df, f"{ticker} Seasonality (10y avg)"), use_container_width=True)
        except Exception as e:
            st.error(f"Seasonality module error: {e}")
    else:
        st.info("No seasonality module found. Computing 10y monthly averages from yfinance.")
        df10 = fetch_price(ticker, period="10y", interval="1mo")
        if df10 is not None and not df10.empty:
            df10["ret"] = df10["Close"].pct_change()
            monthly = df10.groupby(df10.index.month)["ret"].mean()
            monthly.index = pd.to_datetime(monthly.index, format="%m").month_name()
            st.plotly_chart(line_plot(monthly.to_frame("mean_ret"), f"{ticker} Seasonality (10y)"), use_container_width=True)

# -------------------------
# Macro (FRED) tab
# -------------------------
with tabs[4]:
    st.header("Macro - FRED series")
    series_input = fred_series
    codes = [s.strip() for s in series_input.split(",") if s.strip()]
    fred_data = pd.DataFrame()
    if codes:
        for code in codes:
            df = fetch_fred(code, start=fred_start, end=fred_end)
            if not df.empty:
                df.columns = [code]
                if fred_data.empty:
                    fred_data = df
                else:
                    fred_data = fred_data.join(df, how="outer")
        fred_data = fred_data.dropna()
        if not fred_data.empty:
            st.plotly_chart(line_plot(fred_data, "FRED series"), use_container_width=True)
            st.download_button("Download FRED data CSV", fred_data.to_csv().encode(), file_name="fred_data.csv")
        else:
            st.warning("No FRED data available for the series provided.")

# -------------------------
# Bonds tab
# -------------------------
with tabs[5]:
    st.header("Bonds / Rates")
    if bonds_module is not None:
        try:
            bonds_df = bonds_module.get_bond_yields()
            st.plotly_chart(line_plot(bonds_df, "Bond yields"), use_container_width=True)
        except Exception as e:
            st.error(f"Bonds module error: {e}")
    else:
        st.info("No bonds module found. Example: US 10y from FRED (DGS10)")
        dgs10 = fetch_fred("DGS10", start=datetime.now() - pd.DateOffset(years=5))
        if not dgs10.empty:
            st.plotly_chart(line_plot(dgs10, "US 10y yield"), use_container_width=True)

# -------------------------
# TradingView tab
# -------------------------
with tabs[6]:
    st.header("TradingView Widget")
    st.markdown("Embedded TradingView widget. Use a valid symbol like `FX:EURUSD` or `NASDAQ:AAPL`.")
    tv_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div id="tradingview_ea1b2"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
        {{
        "width": "100%",
        "height": 610,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_ea1b2"
      }}
      );
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(tv_html, height=640)

# -------------------------
# Footer / quick exports
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("Quick actions")
if price_df is not None and not price_df.empty:
    csv = price_df.to_csv().encode()
    st.sidebar.download_button("Download price CSV", csv, file_name=f"{ticker.replace('/', '_')}_prices.csv")

st.sidebar.markdown("Built for integration with your existing modules. If functions live in different modules, update import paths at top.")