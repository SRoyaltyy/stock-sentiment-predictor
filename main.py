import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

print('🚀 Stock Sentiment Predictor - 1 Week & 2 Week Forecasts\n')