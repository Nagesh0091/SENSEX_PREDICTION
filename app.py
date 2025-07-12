import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta

# ------------------------ Streamlit Setup ------------------------
st.set_page_config(page_title="SENSEX Forecast", layout="centered")
st.title("📈 SENSEX Forecasting – Live Prediction App")

# ------------------------ Load Trained Models ------------------------
try:
    reg_model = joblib.load("models/reg_model.pkl")
    clf_model = joblib.load("models/clf_model.pkl")
    signal_model = joblib.load("models/signal_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ------------------------ Fetch Live SENSEX Data ------------------------
symbol = "^BSESN"
end_date = datetime.today()
start_date = end_date - timedelta(days=40)

df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")

if df.empty:
    st.error("❌ Couldn't fetch data from Yahoo Finance.")
    st.stop()

df.reset_index(inplace=True)

# ✅ Safe column name formatting (even if tuple comes)
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
df.columns = [str(col).strip().title() for col in df.columns]


# ------------------------ Feature Engineering ------------------------
df['Sma_5'] = df['Close'].rolling(window=5).mean()
df['Sma_10'] = df['Close'].rolling(window=10).mean()
df['Price_Range'] = df['High'] - df['Low']
df['Daily_Change_%'] = ((df['Close'] - df['Open']) / df['Open']) * 100
df['Rolling_Std_5'] = df['Close'].rolling(window=5).std()
df['Close/Open'] = df['Close'] / df['Open']

# ------------------------ Define Features ------------------------
features = [
    'Open', 'High', 'Low', 'Volume',
    'Sma_5', 'Sma_10', 'Price_Range',
    'Daily_Change_%', 'Rolling_Std_5', 'Close/Open'
]

# ------------------------ Drop NA & Check ------------------------
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing columns in data: {missing_cols}")
    st.stop()

df.dropna(subset=features, inplace=True)

if df.empty:
    st.error("❌ Not enough recent data for prediction.")
    st.stop()

# ------------------------ Get Latest Input Row ------------------------
latest_input = df[features].iloc[-1]
latest_date = df['Date'].iloc[-1].strftime("%Y-%m-%d")

# ------------------------ Make Predictions ------------------------
try:
    pred_close = reg_model.predict([latest_input])[0]
    pred_direction = clf_model.predict([latest_input])[0]
    pred_signal = signal_model.predict([latest_input])[0]
except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
    st.stop()


# ------------------------ Decode Predictions ------------------------
direction_text = "📈 Up" if pred_direction == 1 else "📉 Down"
signal_map = {0: "🔻 SELL", 1: "⏸ HOLD", 2: "🔺 BUY"}
signal_text = signal_map.get(pred_signal, "Unknown")

# ------------------------ Display ------------------------
st.markdown(f"### 📅 Latest Market Data Date: **{latest_date}**")
st.dataframe(latest_input.to_frame().rename(columns={latest_input.name: "Value"}))

st.markdown("---")
st.markdown("## 🔮 Prediction for Tomorrow:")
st.success(f"**📊 Predicted Close Price:** ₹{round(pred_close, 2)}")
st.info(f"**🧭 Price Direction:** {direction_text}")
st.warning(f"**📌 Trading Signal:** {signal_text}")

st.markdown("---")
st.caption("📊 Model predictions based on latest SENSEX market data using machine learning. Always do your own analysis before trading.")


# ------------------------ User-friendly Explanation ------------------------
st.markdown("---")
st.markdown("### ℹ️ How to Interpret the Above Prediction")

st.info(
    f"""
    📅 **Latest Market Data Date:** `{latest_date}`  
    ✅ The model has used this latest data to predict for the **very next trading day**.

    Based on the features from {latest_date} (Open, High, Low, Close, Volume, etc.), the model gives:
    
    - 📉 **Predicted Close Price** for the next trading day  
    - 🔁 **Price Direction**: Whether the market might go up or down  
    - 📊 **Trading Signal**: Suggested action (BUY, SELL, HOLD)
    
    This helps you make an informed trading plan for the next market session.
    """
)

st.caption("⚠️ These predictions are for learning/demo purposes. Always verify with your own analysis before trading.")
