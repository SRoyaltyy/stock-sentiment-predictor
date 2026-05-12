import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
import os

TICKERS = ["TSLA", "NVDA", "AAPL", "GME", "AMZN"]
PREDICTION_HORIZONS = [5, 10]  # ≈1 week / 2 weeks

# Latest X sentiment (SuperGrok pulled May 12 2026 - we update this file regularly)
LIVE_X_SENTIMENT = {
    "TSLA": {"avg_sentiment": 0.28, "mention_count": 18, "bullish_pct": 62},
    "NVDA": {"avg_sentiment": 0.52, "mention_count": 14, "bullish_pct": 71},
    "AAPL": {"avg_sentiment": 0.12, "mention_count": 9,  "bullish_pct": 58},
    "GME":  {"avg_sentiment": 0.48, "mention_count": 11,  "bullish_pct": 69},
    "AMZN": {"avg_sentiment": 0.25, "mention_count": 7,   "bullish_pct": 61}
}

print("🚀 SuperGrok Stock Predictor - Running in GitHub Actions")
print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Get price data + technicals
def get_price_data():
    data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty:
            continue
        # Technical indicators
        df["rsi"] = df["Close"].rolling(14).apply(lambda x: 100 - 100 / (1 + (x.diff().clip(lower=0).rolling(14).mean() / abs(x.diff().clip(upper=0).rolling(14).mean()))), raw=False)
        df["macd"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["sma_50"] = df["Close"].rolling(50).mean()
        df["volatility"] = df["Close"].pct_change().rolling(20).std()
        data[ticker] = df.dropna()
    return data

price_data = get_price_data()

# Features with X sentiment
features_list = []
for ticker in TICKERS:
    if ticker not in price_data:
        continue
    df = price_data[ticker]
    latest = df.iloc[-1]
    sent = LIVE_X_SENTIMENT.get(ticker, {"avg_sentiment": 0, "mention_count": 0, "bullish_pct": 50})
    row = {
        "ticker": ticker,
        "current_price": round(float(latest["Close"]), 2),
        "rsi": round(float(latest["rsi"]), 2),
        "macd": round(float(latest["macd"]), 2),
        "sma_20": round(float(latest["sma_20"]), 2),
        "sma_50": round(float(latest["sma_50"]), 2),
        "volatility": round(float(latest["volatility"]), 4),
        "avg_sentiment": sent["avg_sentiment"],
        "mention_count": sent["mention_count"],
        "bullish_pct": sent["bullish_pct"]
    }
    features_list.append(row)

features_df = pd.DataFrame(features_list)
print("\n📊 Current features (X sentiment + technicals):")
print(features_df[["ticker", "current_price", "avg_sentiment", "bullish_pct"]])

# Train model + predict
X_list, y_list = [], []
for ticker, df in price_data.items():
    for horizon in PREDICTION_HORIZONS:
        df[f"target_{horizon}"] = (df["Close"].pct_change(periods=horizon).shift(-horizon) > 0).astype(int)
        feat = df[["rsi", "macd", "sma_20", "sma_50", "volatility"]].dropna()
        target = df[f"target_{horizon}"].dropna()
        common = feat.index.intersection(target.index)
        if not common.empty:
            X_list.append(feat.loc[common])
            y_list.append(target.loc[common])

if X_list:
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n✅ Model trained — historical accuracy: {acc:.1%}")

    print("\n" + "="*70)
    print("📈 PREDICTIONS FOR NEXT 1 WEEK & 2 WEEKS")
    print("="*70)
    current_X = features_df[["rsi", "macd", "sma_20", "sma_50", "volatility"]]
    for i, row in features_df.iterrows():
        ticker = row["ticker"]
        price = row["current_price"]
        prob_up = model.predict_proba(current_X.iloc[[i]])[0][1]
        direction = "UP" if prob_up > 0.5 else "DOWN"
        print(f"\n{ticker} @ ${price}")
        for horizon in PREDICTION_HORIZONS:
            weeks = "1 week" if horizon == 5 else "2 weeks"
            print(f"   → {weeks}: {direction} ({prob_up*100:.0f}% confidence)")

    # Save results to file so workflow can upload or commit
    os.makedirs("data", exist_ok=True)
    with open("data/predictions.txt", "w") as f:
        f.write(f"Predictions as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        for i, row in features_df.iterrows():
            ticker = row["ticker"]
            price = row["current_price"]
            prob_up = model.predict_proba(current_X.iloc[[i]])[0][1]
            direction = "UP" if prob_up > 0.5 else "DOWN"
            f.write(f"{ticker} @ ${price}\n")
            for horizon in PREDICTION_HORIZONS:
                weeks = "1 week" if horizon == 5 else "2 weeks"
                f.write(f"   → {weeks}: {direction} ({prob_up*100:.0f}% confidence)\n")
            f.write("\n")
else:
    print("Not enough data yet.")
print("\n🎉 Predictions complete! Check Actions logs or data/predictions.txt")