import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load CSV
data = pd.read_csv("stock.csv", parse_dates=["Date"])
data.set_index("Date", inplace=True)

# Use Adjusted Close (best for stocks)
series = data["Adj Close"].values.reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series)

def create_dataset(series, time_step=10):
    X, y = [], []
    for i in range(len(series) - time_step):
        X.append(series[i:i + time_step, 0])
        y.append(series[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 20
X, y = create_dataset(scaled_series, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Forecast future values
future_days = 5

last_window = scaled_series[-time_step:].reshape(1, time_step, 1)
future_preds = [] 

for _ in range(future_days):
    next_price = model.predict(last_window, verbose=0)
    future_preds.append(next_price[0, 0])

    last_window = np.roll(last_window, -1, axis=1)
    last_window[0, -1, 0] = next_price

future_preds = scaler.inverse_transform(
    np.array(future_preds).reshape(-1, 1)
)

print("Future prices:")
for i, p in enumerate(future_preds, 1):
    print(f"Day +{i}: {p[0]:.2f}")