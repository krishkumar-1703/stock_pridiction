import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from features import build_features
from forecast import download_live, drop_incomplete_day, load_models, make_forecast
from metrics import directional_accuracy

st.set_page_config(page_title="Nifty 50 forecasting prototype", layout="wide")

DISCLAIMER = ("Educational / research prototype. Model estimates only, not financial advice. "
              "Past patterns do not guarantee future results.")
RANGES = {"6 months": 126, "1 year": 252, "3 years": 756, "5 years": 1260, "All": None}


@st.cache_data(ttl=3600, show_spinner="Downloading latest daily data...")
def get_live_data():
    return download_live()


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return load_models()


@st.cache_data
def load_dated(path):
    return pd.read_csv(path, index_col="Date", parse_dates=True)


@st.cache_data
def load_table(path):
    return pd.read_csv(path, index_col=0)


def get_data():
    try:
        df, source = get_live_data(), "live download (yfinance)"
    except Exception as err:
        df = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)
        source = f"FROZEN SNAPSHOT (live download failed: {type(err).__name__})"
    df, dropped = drop_incomplete_day(df)
    return df, source, dropped


def fmt(value, pattern):
    return "n/a" if pd.isna(value) else pattern.format(value)


# ---------------- load everything ----------------
df, source, dropped = get_data()
feats_full = build_features(df)
test = load_dated(str(config.TEST_PATH))
train = load_dated(str(config.TRAIN_PATH))
preds = load_dated(str(config.RESULTS_DIR / "test_predictions.csv"))
train_end = train.index.max().date()

# ---------------- header and sidebar ----------------
st.title("Nifty 50 next-day forecasting: baselines vs machine learning vs deep learning")
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Instrument")
    st.write(f"**Nifty 50** (`{config.TICKER}`)")
    st.caption("The models were trained on this index only, so no other tickers are offered.")
    st.write(f"Data source: {source}")
    st.write(f"Last complete trading day: **{df.index[-1].date()}**")
    if dropped:
        st.caption("Today's session may still be running, so today's row is excluded.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Models were trained on data up to {train_end} and have not been retrained since.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price and indicators", "Test-period predictions", "Error metrics", "Next-day forecast", "Limitations"])

# ---------------- tab 1: price and indicators ----------------
with tab1:
    c1, c2 = st.columns(2)
    span = c1.radio("Range", list(RANGES), index=2, horizontal=True)
    mas = c2.multiselect("Moving averages (days)", [5, 10, 20, 50], default=[20, 50])
    n = RANGES[span] or len(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    view = df.iloc[-n:]
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", mode="lines"), row=1, col=1)
    for w in mas:
        sma = df["Close"].rolling(w).mean().iloc[-n:]
        fig.add_trace(go.Scatter(x=sma.index, y=sma.values, name=f"SMA {w}", mode="lines"), row=1, col=1)
    rsi = feats_full["rsi_14"].iloc[-n:]
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name="RSI (14)", mode="lines"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_yaxes(title_text="Index level", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig)
    st.caption("Indicators are shown for context only. The forecasting models do not use raw prices: "
               "they use relative features (returns, price-to-average ratios, volatility, RSI).")

# ---------------- tab 2: test-period predictions ----------------
with tab2:
    st.write(f"Held-out test period: **{test.index.min().date()}** to **{test.index.max().date()}** "
             f"({len(test)} trading days). The models were trained and tuned on earlier data only.")
    columns = {"Naive": "naive", "Moving average": "moving_avg", "Linear Regression": "linear_reg",
               "Random Forest": f"rf_s{config.RF_SEEDS[0]}", "LSTM": f"lstm_s{config.LSTM_SEEDS[0]}"}
    c1, c2 = st.columns(2)
    chosen = c1.multiselect("Models", list(columns), default=["Naive", "Random Forest", "LSTM"])
    show_what = c2.radio("Show", ["Next-day close price", "Next-day return (%)"], horizontal=True)
    last_n = st.slider("Number of most recent test days", 30, len(test), 60)

    t = test.iloc[-last_n:]
    p = preds.loc[t.index]
    fig = go.Figure()
    if show_what.startswith("Next-day close"):
        fig.add_trace(go.Scatter(x=t.index, y=t["next_close"], name="Actual", mode="lines+markers",
                                 line=dict(width=3)))
        for name in chosen:
            fig.add_trace(go.Scatter(x=t.index, y=t["close"] * (1 + p[columns[name]]), name=name, mode="lines"))
        fig.update_yaxes(title_text="Next-day close")
    else:
        fig.add_trace(go.Bar(x=t.index, y=t[config.TARGET_COL] * 100, name="Actual"))
        for name in chosen:
            fig.add_trace(go.Scatter(x=t.index, y=p[columns[name]] * 100, name=name, mode="lines"))
        fig.update_yaxes(title_text="Next-day return (%)")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig)
    st.caption("The x-axis is the day each forecast was made; the lines show what the forecast said about the "
               "following day. Random Forest and LSTM are the main (first-seed) models.")
    st.caption("On the price view every model looks almost identical to 'copy yesterday', which is why "
               "price-level R-squared is very high for all of them. Switch to the return view to see the "
               "real difference: forecasts are tiny compared with what actually happens.")

# ---------------- tab 3: error metrics ----------------
with tab3:
    mean_tbl = load_table(str(config.RESULTS_DIR / "final_test_table.csv"))
    std_tbl = load_table(str(config.RESULTS_DIR / "final_test_table_seed_std.csv"))
    sig = load_table(str(config.RESULTS_DIR / "significance_vs_naive.csv"))

    st.subheader("Test-period results (all numbers come from the saved experiment run)")
    show = pd.DataFrame({
        "RMSE (index points)": mean_tbl["rmse_price"].map(lambda v: fmt(v, "{:,.2f}")),
        "MAE (index points)": mean_tbl["mae_price"].map(lambda v: fmt(v, "{:,.2f}")),
        "MAPE (%)": mean_tbl["mape_price"].map(lambda v: fmt(v, "{:.3f}")),
        "Directional accuracy (%)": mean_tbl["dir_acc"].map(lambda v: fmt(v, "{:.1f}")),
        "RMSE (return)": mean_tbl["rmse_ret"].map(lambda v: fmt(v, "{:.6f}")),
        "R-squared (return)": mean_tbl["r2_ret"].map(lambda v: fmt(v, "{:.4f}")),
    })
    st.dataframe(show)
    st.caption("Random Forest and LSTM rows are the mean over 5 random seeds. Price-level errors are close for "
               "every model because tomorrow's price is mostly today's price; the return-level columns are the "
               "fairer comparison.")

    y = test[config.TARGET_COL].values
    up = directional_accuracy(y, np.ones(len(test)))
    same = directional_accuracy(y, test["ret_lag0"].values)
    noise = 1.96 * np.sqrt(0.25 / len(test)) * 100
    st.write(f"Direction references on the test period: always-up **{up:.1f}%**, "
             f"same-direction-as-today **{same:.1f}%**. With {len(test)} days, a direction score has sampling "
             f"noise of about +/- {noise:.1f} points, so small gaps are not evidence of skill.")

    st.subheader("Stability across random seeds")
    rows = [n for n in std_tbl.index if n.startswith(("Random Forest", "LSTM"))]
    spread = pd.DataFrame({
        "Std of return RMSE": std_tbl.loc[rows, "rmse_ret"].map("{:.6f}".format),
        "Std of directional accuracy (points)": std_tbl.loc[rows, "dir_acc"].map("{:.2f}".format),
    })
    st.dataframe(spread)

    st.subheader("Is any model really better than 'tomorrow = today'?")
    sig_show = pd.DataFrame({
        "Squared-error change vs naive (%)": sig["mse_reduction_pct"].map("{:+.2f}".format),
        "95% interval (%)": [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(sig["ci95_low_pct"], sig["ci95_high_pct"])],
    })
    st.dataframe(sig_show)
    st.caption("Positive means better than naive. Paired block bootstrap on the test days. "
               "An interval that includes 0 cannot be told apart from chance.")
    better = sig.index[sig["ci95_low_pct"] > 0].tolist()
    worse = sig.index[sig["ci95_high_pct"] < 0].tolist()
    if better:
        st.success("Better than naive with 95% confidence: " + ", ".join(better))
    else:
        st.info("No model's 95% interval lies entirely above zero, so none is shown to beat the naive "
                "baseline on this test period.")
    if worse:
        st.warning("Clearly worse than naive: " + ", ".join(worse))

# ---------------- tab 4: next-day forecast ----------------
with tab4:
    if "FROZEN" in source:
        st.warning("The live download failed, so this forecast uses the frozen snapshot, not the latest data.")
    models = get_models()
    asof, close_t, table = make_forecast(df, models)
    names = list(table.index)
    st.write(f"Forecast made from the close of **{asof.date()}** (close {close_t:,.2f}) "
             f"for the **next trading session**.")
    choice = st.selectbox("Model", names, index=names.index("Random Forest"))
    row = table.loc[choice]
    typical = float(test[config.TARGET_COL].std())

    c1, c2 = st.columns(2)
    c1.metric("Model estimate: next-session close", f"{row['pred_close']:,.2f}",
              f"{row['pred_return'] * 100:+.3f}% vs last close", delta_color="off")
    c2.metric("Typical daily move in the test period", f"{typical * 100:.2f}%")
    st.caption(f"This estimate is {abs(row['pred_return']) / typical * 100:.0f}% of a typical daily move "
               f"(the standard deviation of daily returns in the test period).")

    st.write("All models:")
    shown = pd.DataFrame({
        "Estimated return (%)": (table["pred_return"] * 100).map("{:+.3f}".format),
        "Estimated close": table["pred_close"].map("{:,.2f}".format),
    })
    st.dataframe(shown)

    signs = np.sign(table["pred_return"].drop("Naive"))
    if signs.nunique() > 1:
        st.info("The models do not agree on the direction of the next move.")
    else:
        st.info("The models point the same way, but see the Error metrics tab: their directional accuracy on "
                "the test period was not clearly better than chance.")

    st.warning("Model estimate, not advice. " + DISCLAIMER)
    st.caption(f"These models were trained on data up to {train_end} and not retrained on newer data. "
               f"The Error metrics tab shows how each model did on a held-out period, including whether it "
               f"beat the naive baseline.")

# ---------------- tab 5: limitations ----------------
with tab5:
    st.markdown("""
**Limitations of this project**

- Markets are noisy and close to efficient, so predictability is limited.
- Historical patterns do not guarantee future performance.
- Models can overfit and can fail in new market regimes.
- The test period was flat and relatively calm, with no crash, so it cannot show how the models behave in a crisis.
- News, macro events and sentiment are not captured.
- Prediction accuracy is not trading profit: transaction costs and execution are outside this project.
- The data source (yfinance) is unofficial and can have gaps or errors.
- A good backtest or forecast is not proof of future accuracy or profit.

**This is an educational / research prototype, not financial advice.**
""")

st.caption(DISCLAIMER)