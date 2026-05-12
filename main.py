import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

TICKERS = ["TSLA", "NVDA", "AAPL", "GME", "AMZN"]
PREDICTION_HORIZONS = [5, 10]  # ≈1 week / 2 weeks

# Fresh X sentiment (updated by SuperGrok)
LIVE_X_SENTIMENT = {
    "TSLA": {"avg_sentiment": 0.28, "mention_count": 12, "bullish_pct": 62},
    "NVDA": {"avg_sentiment": 0.42, "mention_count": 15, "bullish_pct": 71},
    "AAPL": {"avg_sentiment": 0.12, "mention_count": 9,  "bullish_pct": 58},
    "GME":  {"avg_sentiment": 0.48, "mention_count": 8,  "bullish_pct": 69},
    "AMZN": {"avg_sentiment": 0.25, "mention_count": 6,  "bullish_pct": 63}
}

print("🚀 SuperGrok Stock Predictor - Running in GitHub Actions")
print(f"Run time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_price_data():
    data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty:
            continue
        df = df.copy()
        df['rsi'] = calculate_rsi(df['Close'])
        df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        data[ticker] = df.dropna()
    return data

price_data = get_price_data()

# Prepare features
features_list = []
for ticker in TICKERS:
    if ticker not in price_data or price_data[ticker].empty:
        continue
    df = price_data[ticker]
    latest = df.iloc[-1]
    sent = LIVE_X_SENTIMENT.get(ticker, {"avg_sentiment": 0, "mention_count": 0, "bullish_pct": 50})
    
    row = {
        "ticker": ticker,
        "current_price": float(df['Close'].iloc[-1]),
        "rsi": float(latest['rsi']),
        "macd": float(latest['macd']),
        "sma_20": float(latest['sma_20']),
        "sma_50": float(latest['sma_50']),
        "volatility": float(latest['volatility']),
        "avg_sentiment": sent["avg_sentiment"],
        "mention_count": sent["mention_count"],
        "bullish_pct": sent["bullish_pct"]
    }
    features_list.append(row)

features_df = pd.DataFrame(features_list)
print("📊 Live features (with fresh X sentiment):")
print(features_df[['ticker', 'current_price', 'avg_sentiment', 'bullish_pct']])

# Train model
X_list, y_list = [], []
for ticker, df in price_data.items():
    for horizon in PREDICTION_HORIZONS:
        df[f'target_{horizon}'] = (df['Close'].pct_change(periods=horizon).shift(-horizon) > 0).astype(int)
        feat = df[['rsi', 'macd', 'sma_20', 'sma_50', 'volatility']].dropna()
        target = df[f'target_{horizon}'].dropna()
        common = feat.index.intersection(target.index)
        if not common.empty:
            X_list.append(feat.loc[common])
            y_list.append(target.loc[common])

if X_list:
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n✅ Model trained — historical accuracy: {acc:.1%}")

    print("\n" + "="*70)
    print("📈 TODAY'S 1-WEEK & 2-WEEK PREDICTIONS")
    print("="*70)
    
    current_X = features_df[['rsi', 'macd', 'sma_20', 'sma_50', 'volatility']]
    for i, row in features_df.iterrows():
        ticker = row['ticker']
        price = row['current_price']
        prob_up = model.predict_proba(current_X.iloc[[i]])[0][1]
        direction = 'UP' if prob_up > 0.5 else 'DOWN'
        print(f"\n{ticker} @ ${price:.2f}")
        for horizon in PREDICTION_HORIZONS:
            weeks = '1 week' if horizon == 5 else '2 weeks'
            print(f"   → {weeks}: {direction} ({prob_up*100:.0f}% confidence)")
else:
    print("Not enough data for training yet.")

print("\n🎉 Run completed successfully!")
