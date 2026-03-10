import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention
from sklearn.preprocessing import MinMaxScaler

TIME_STEP = 20
FORECAST_HORIZON = 3

# ------------------------------
# Load stock data
# ------------------------------

stock_data = pd.read_csv("stock.csv", parse_dates=["Date"], index_col="Date")

features = ["Adj Close","Close","High","Low","Open","Volume"]
stock_data = stock_data[features]

# ------------------------------
# Load news data
# ------------------------------

news_data = pd.read_csv("daily_features.csv", parse_dates=["date"], index_col="date")

news_features = ["net_impact","weighted_horizon","news_count","divergence"]
news_data = news_data[news_features]

# ------------------------------
# Align news to stock days
# ------------------------------

news_data = news_data.reindex(stock_data.index).fillna(0)

# ------------------------------
# Scaling
# ------------------------------

stock_scaler = MinMaxScaler()
scaled_stock = stock_scaler.fit_transform(stock_data)

news_scaler = MinMaxScaler()
scaled_news = news_scaler.fit_transform(news_data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(stock_data[["Close"]])

scaled_stock_df = pd.DataFrame(scaled_stock, index=stock_data.index, columns=features)
scaled_news_df = pd.DataFrame(scaled_news, index=news_data.index, columns=news_features)
scaled_close_df = pd.DataFrame(scaled_close, index=stock_data.index, columns=["Close"])

# ------------------------------
# Dataset creation
# ------------------------------

def create_dataset(stock_df, news_df, close_df):

    X_stock, X_news, y = [], [], []

    for i in range(len(stock_df) - TIME_STEP - FORECAST_HORIZON + 1):

        X_stock.append(stock_df.iloc[i:i+TIME_STEP].values)
        X_news.append(news_df.iloc[i:i+TIME_STEP].values)

        target = close_df.iloc[i+TIME_STEP:i+TIME_STEP+FORECAST_HORIZON]
        y.append(target.values.flatten())

    return np.array(X_stock), np.array(X_news), np.array(y)

X_stock, X_news, y = create_dataset(
    scaled_stock_df,
    scaled_news_df,
    scaled_close_df
)

# ------------------------------
# Model
# ------------------------------

# STOCK BRANCH
stock_input = Input(shape=(TIME_STEP, 6))

x_stock = LSTM(64, return_sequences=True)(stock_input)
x_stock = LSTM(64)(x_stock)

# NEWS BRANCH
news_input = Input(shape=(TIME_STEP, 4))

masked_news = Masking(mask_value=0.0)(news_input)

x_news = Dense(64, activation="relu")(masked_news)

# Attention over news timeline
attention_layer = Attention()
news_context = attention_layer([x_news, x_news])

# Collapse time dimension
news_vector = LSTM(32)(news_context)

# MERGE
merged = Concatenate()([x_stock, news_vector])

merged = Dense(64, activation="relu")(merged)
merged = Dense(32, activation="relu")(merged)

output = Dense(FORECAST_HORIZON)(merged)

model = Model(inputs=[stock_input, news_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# ------------------------------
# Training
# ------------------------------

model.fit(
    [X_stock, X_news],
    y,
    epochs=150,
    batch_size=32
)

# ------------------------------
# Forecast
# ------------------------------

last_stock = stock_data.iloc[-TIME_STEP:]
last_news = news_data.iloc[-TIME_STEP:]

last_stock_scaled = stock_scaler.transform(last_stock)
last_news_scaled = news_scaler.transform(last_news)

last_stock_scaled = last_stock_scaled.reshape(1, TIME_STEP, 6)
last_news_scaled = last_news_scaled.reshape(1, TIME_STEP, 4)

pred_scaled = model.predict([last_stock_scaled, last_news_scaled])[0]

pred_scaled = np.clip(pred_scaled, 0, 1)

predicted_close = close_scaler.inverse_transform(
    pred_scaled.reshape(-1,1)
).flatten()

print("\nNext 3 day closing prices:")

for i, price in enumerate(predicted_close, 1):
    print(f"Day +{i}: {price:.2f}")