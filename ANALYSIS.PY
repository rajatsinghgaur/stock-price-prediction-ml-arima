import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("reliance_raw_historical.csv")
data.head()
print("Original Shape:", data.shape)
print(data.head())

#data cleaning
import pandas as pd
import numpy as np

# Load Data
data = pd.read_csv("reliance_raw_historical.csv")

print("Original Shape:", data.shape)

# Clean Column Names
data.columns = data.columns.str.strip()

# Convert Date
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])

# Sort
data = data.sort_values('Date')

# Remove Duplicates
data = data.drop_duplicates()

# Convert Numeric Columns
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Forward Fill (NEW WAY)
data = data.ffill()

# Drop Remaining NaN
data = data.dropna()

# Logical Check
data = data[data['High'] >= data['Low']]

# Set Index
data.set_index('Date', inplace=True)

print("Cleaned Shape:", data.shape)
print(data.head())

#EDA

# EXPLORATORY DATA ANALYSIS

import matplotlib.pyplot as plt
import numpy as np

print("Data Shape:", data.shape)
print("\nData Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.title("Reliance Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(data['Volume'])
plt.title("Trading Volume Trend")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.show()

data['Daily_Return'] = data['Close'].pct_change()
print("\nDaily Returns Summary;")
print(data['Daily_Return'].describe())

plt.figure(figsize=(12,6))
plt.plot(data['Daily_Return'])
plt.title("Daily Returns")
plt.grid(True)
plt.show()

data['Volatility'] = data['Daily_Return'].rolling(20).std()
print("\nVolatility Summary:")
print(data['Volatility'].describe())
print("\nVolatility Head;")
print(data['Volatility'].head())

plt.figure(figsize=(12,6))
plt.plot(data['Volatility'])
plt.title("20-Day Rolling Volatility")
plt.grid(True)
plt.show()

data['MA_10'] = data['Close'].rolling(10).mean()
data['MA_50'] = data['Close'].rolling(50).mean()
print("\nMoving Averages Summary;")
print(data[['MA_10','MA_50']].describe())


plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close')
plt.plot(data['MA_10'], label='MA 10')
plt.plot(data['MA_50'], label='MA 50')
plt.legend()
plt.title("Moving Averages")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.imshow(data.corr(), cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(10,5))
plt.hist(data['Daily_Return'], bins=50)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ==============================
# BOXPLOT – CLOSE PRICE
# ==============================

plt.figure(figsize=(8,5))

plt.boxplot(data['Close'], vert=False)

plt.title("Boxplot of Close Price")
plt.xlabel("Close Price")
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))

plt.boxplot(data['Daily_Return'], vert=False)

plt.title("Boxplot of Daily Returns")
plt.xlabel("Daily Return")
plt.grid(True)

plt.show()

plt.figure(figsize=(8,6))

corr = data.corr()

plt.imshow(corr)
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Matrix")

plt.show()

# Moving Averages
data['MA_10'] = data['Close'].rolling(10).mean()
data['MA_50'] = data['Close'].rolling(50).mean()

print("="*50)
print("Moving Average Values (Last 5 Rows)")
print(data[['Close','MA_10','MA_50']].tail())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(data['Close'], label="Close Price")
plt.plot(data['MA_10'], label="MA 10")
plt.plot(data['MA_50'], label="MA 50")
plt.legend()
plt.title("Moving Average Indicator")
plt.show()

delta = data['Close'].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

print("="*50)
print("RSI Values (Last 5 Rows)")
print(data[['Close','RSI']].tail())

plt.figure(figsize=(12,6))
plt.plot(data['RSI'], label="RSI")
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='g', linestyle='--')
plt.title("RSI Indicator")
plt.legend()
plt.show()

exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()

data['MACD'] = exp1 - exp2

print("="*50)
print("MACD Values (Last 5 Rows)")
print(data[['Close','MACD']].tail())


plt.figure(figsize=(12,6))
plt.plot(data['MACD'], label="MACD")
plt.axhline(0, color='black')
plt.title("MACD Indicator")
plt.legend()
plt.show()

# Daily Return
data['Daily_Return'] = data['Close'].pct_change()

print("="*50)
print("Daily Return (Last 5 Rows)")
print(data[['Close','Daily_Return']].tail())

# Previous Day Close
data['Lag_1'] = data['Close'].shift(1)

print("="*50)
print("Lag Feature (Last 5 Rows)")
print(data[['Close','Lag_1']].tail())

# 20 Day Rolling Volatility
data['Volatility'] = data['Daily_Return'].rolling(20).std()

print("="*50)
print("Volatility (Last 5 Rows)")
print(data[['Volatility']].tail())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(data['Volatility'])
plt.title("Rolling 20 Day Volatility")
plt.show()

# Daily Price Range
data['Price_Range'] = data['High'] - data['Low']

print("="*50)
print("Price Range (Last 5 Rows)")
print(data[['Price_Range']].tail())

data['Volume_Change'] = data['Volume'].pct_change()

print("="*50)
print("Volume Change (Last 5 Rows)")
print(data[['Volume_Change']].tail())

data.dropna(inplace=True)

print("="*50)
print("Final Data Shape After Feature Engineering:")
print(data.shape)

features = ['Open','High','Low','Volume',
            'MA_10','MA_50',
            'RSI','MACD',
            'Lag_1','Volatility',
            'Price_Range']

print("="*50)
print("Final Features Used:")
print(features)

# Define Features & Target
X = data[['Open','High','Low','Volume',
          'MA_10','MA_50','RSI','MACD',
          'Lag_1','Volatility','Price_Range']]

y = data['Close']

train_size = int(len(data) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

pred_gb = gb.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("="*50)
    print("Model:", name)
    print("R2 Score:", round(r2,4))
    print("RMSE:", round(rmse,4))
    
    return r2, rmse

r2_lr, rmse_lr = evaluate_model("Linear Regression", y_test, pred_lr)
r2_rf, rmse_rf = evaluate_model("Random Forest", y_test, pred_rf)
r2_gb, rmse_gb = evaluate_model("Gradient Boosting", y_test, pred_gb)

print("="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

print("Linear Regression  | R2:", round(r2_lr,4), "| RMSE:", round(rmse_lr,4))
print("Random Forest      | R2:", round(r2_rf,4), "| RMSE:", round(rmse_rf,4))
print("Gradient Boosting  | R2:", round(r2_gb,4), "| RMSE:", round(rmse_gb,4))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual")
plt.plot(pred_rf, label="Random Forest Prediction")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

result = adfuller(data['Close'])

print("="*50)
print("ADF TEST RESULT")
print("="*50)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

data['Close_diff'] = data['Close'].diff()

print("After Differencing - Check First 5 Values:")
print(data[['Close','Close_diff']].head())

data.dropna(inplace=True)

model = ARIMA(data['Close'], order=(5,1,0))
result = model.fit()

print("="*50)
print("ARIMA MODEL SUMMARY")
print("="*50)
print(result.summary())

forecast = result.forecast(steps=30)

print("="*50)
print("Next 30 Days Forecast")
print("="*50)
print(forecast)

plt.figure(figsize=(12,6))

plt.plot(data['Close'], label="Historical Price")
plt.plot(range(len(data), len(data)+30), forecast, label="Forecast")

plt.legend()
plt.title("ARIMA Forecast (Next 30 Days)")
plt.show()

# Daily Return
data['Daily_Return'] = data['Close'].pct_change()

print("="*50)
print("Daily Return Summary")
print("="*50)
print(data['Daily_Return'].describe())

import numpy as np

volatility = np.std(data['Daily_Return']) * np.sqrt(252)

print("="*50)
print("Annualized Volatility:")
print(round(volatility,4))

data['Rolling_Volatility'] = data['Daily_Return'].rolling(20).std()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(data['Rolling_Volatility'])
plt.title("20-Day Rolling Volatility")
plt.show()

VaR_95 = np.percentile(data['Daily_Return'].dropna(), 5)

print("="*50)
print("Value at Risk (95% Confidence):")
print(round(VaR_95,4))

data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
data['Cumulative_Max'] = data['Cumulative_Return'].cummax()

data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1

max_drawdown = data['Drawdown'].min()

print("="*50)
print("Maximum Drawdown:")
print(round(max_drawdown,4))

plt.figure(figsize=(12,6))
plt.plot(data['Drawdown'])
plt.title("Drawdown Over Time")
plt.show()

mean_return = data['Daily_Return'].mean() * 252

print("="*50)
print("Annualized Return:", round(mean_return,4))
print("Annualized Volatility:", round(volatility,4))

# Create Signal Column
data['Signal'] = 0

data.loc[data['MA_10'] > data['MA_50'], 'Signal'] = 1
data.loc[data['MA_10'] < data['MA_50'], 'Signal'] = -1

print("="*50)
print("Signal Distribution:")
print(data['Signal'].value_counts())

# Strategy Return
data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']

print("="*50)
print("Strategy Return Summary:")
print(data['Strategy_Return'].describe())

data['Cumulative_Market_Return'] = (1 + data['Daily_Return']).cumprod()
data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

print("="*50)
print("Final Strategy Value:")
print(data[['Cumulative_Market_Return','Cumulative_Strategy_Return']].tail())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(data['Cumulative_Market_Return'], label="Market Return")
plt.plot(data['Cumulative_Strategy_Return'], label="Strategy Return")

plt.legend()
plt.title("Backtesting: Strategy vs Market")
plt.show()

market_return = data['Cumulative_Market_Return'].iloc[-1] - 1
strategy_return = data['Cumulative_Strategy_Return'].iloc[-1] - 1

print("="*50)
print("Final Market Return:", round(market_return,4))
print("Final Strategy Return:", round(strategy_return,4))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# =====================================================
# 1. LOAD DATA
# =====================================================

data = pd.read_csv("reliance_raw_historical.csv")

# Convert Date column properly
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])

data = data.sort_values('Date')
data.set_index('Date', inplace=True)

# REMOVE TIMEZONE IF EXISTS (Fixes your error)
if data.index.tz is not None:
    data.index = data.index.tz_localize(None)

print("="*60)
print("Last Available Date in Dataset:", data.index[-1])
print("="*60)

# =====================================================
# 2. TRAIN ARIMA MODEL
# =====================================================

print("\nTraining ARIMA Model...\n")

model = ARIMA(data['Close'], order=(5,1,0))
result = model.fit()

print("Model Training Complete")

# =====================================================
# 3. FORECAST UNTIL MARCH 31, 2026
# =====================================================

target_date = pd.to_datetime("2026-03-31")
last_date = data.index[-1]

# Calculate forecast steps
steps = (target_date - last_date).days

if steps <= 0:
    print("Target date already within dataset range.")
else:
    print("\nForecast Steps Required:", steps)

    forecast = result.forecast(steps=steps)

    # Create future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=steps)

    forecast_series = pd.Series(forecast.values, index=future_dates)

    # =====================================================
    # 4. PRINT ONLY MARCH 2026 VALUES
    # =====================================================

    print("\n" + "="*60)
    print("FORECASTED PRICES FOR MARCH 2026")
    print("="*60)

    march_forecast = forecast_series["2026-03-01":"2026-03-31"]

    print(march_forecast)

    print("\nAverage Predicted Price in March 2026:",
          round(march_forecast.mean(),2))

    print("Predicted Closing Price on 31 March 2026:",
          round(march_forecast.iloc[-1],2))

    # =====================================================
    # 5. PLOT
    # =====================================================

    plt.figure(figsize=(12,6))

    plt.plot(data['Close'], label="Historical Price")
    plt.plot(forecast_series, label="Forecast till March 31, 2026")

    plt.title("Stock Price Forecast till March 31, 2026")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

input("\nPress Enter to Exit...")
