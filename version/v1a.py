import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
TIME_STEP = 40
FORECAST_HORIZON = 3

# ------------------------------
# Load stock data
# ------------------------------

stock_data = pd.read_csv("data/raw/stock.csv", parse_dates=["Date"], index_col="Date")

features = ["Adj Close","Close","High","Low","Open","Volume"]
stock_data = stock_data[features]

# ------------------------------
# Load news data
# ------------------------------

news_data = pd.read_csv("data/processed/daily_features.csv", parse_dates=["date"], index_col="date")

news_features = ["net_impact","weighted_horizon","news_count","divergence"]
news_data = news_data[news_features]

#------------------------------
# Load PHLX Semiconductor Sector (SOX) index data
#------------------------------
sox_data = pd.read_csv("data/raw/^SOX.csv", parse_dates=["Date"], index_col="Date")
sox_data = sox_data[["Adj Close"]]

# ------------------------------
# Align news to stock days
# ------------------------------
news_data = news_data.reindex(stock_data.index).fillna(0)
sox_data = sox_data.reindex(stock_data.index).ffill().bfill()
sox_data = sox_data.shift(1).bfill()
# ------------------------------
# Scaling
# ------------------------------
split_idx = int(len(stock_data) * 0.7)
train_size = split_idx - TIME_STEP - FORECAST_HORIZON + 1

train_stock = stock_data.iloc[:split_idx]
train_news = news_data.iloc[:split_idx]
train_sox = sox_data.iloc[:split_idx]


stock_scaler = MinMaxScaler()
stock_scaler.fit(train_stock)
scaled_stock = stock_scaler.transform(stock_data)

news_scaler = MinMaxScaler()
news_scaler.fit(train_news)
scaled_news = news_scaler.transform(news_data)

sox_scaler = MinMaxScaler()
sox_scaler.fit(train_sox)
scaled_sox = sox_scaler.transform(sox_data)

close_scaler = MinMaxScaler()
close_scaler.fit(train_stock[["Adj Close"]])
scaled_close = close_scaler.transform(stock_data[["Adj Close"]])

scaled_stock_df = pd.DataFrame(scaled_stock, index=stock_data.index, columns=features)
scaled_news_df = pd.DataFrame(scaled_news, index=news_data.index, columns=news_features)
scaled_sox_df = pd.DataFrame(scaled_sox, index=sox_data.index, columns=["Adj Close"])
scaled_close_df = pd.DataFrame(scaled_close, index=stock_data.index, columns=["Adj Close"])

# ------------------------------
# Dataset creation
# ------------------------------

def create_dataset(stock_df, news_df, sox_df, close_df):

    X_stock, X_news, X_sox, y = [], [], [], []

    for i in range(len(stock_df) - TIME_STEP - FORECAST_HORIZON + 1):

        X_stock.append(stock_df.iloc[i:i+TIME_STEP].values)
        X_news.append(news_df.iloc[i:i+TIME_STEP].values)
        X_sox.append(sox_df.iloc[i:i+TIME_STEP].values)
        target = close_df.iloc[i+TIME_STEP:i+TIME_STEP+FORECAST_HORIZON]
        y.append(target.values.flatten())

    return np.array(X_stock), np.array(X_news), np.array(X_sox), np.array(y)

X_stock, X_news, X_sox, y = create_dataset(
    scaled_stock_df,
    scaled_news_df,
    scaled_sox_df,
    scaled_close_df
)

# ------------------------------
# Model
# ------------------------------

# STOCK BRANCH
stock_input = Input(shape=(TIME_STEP, 6), name="stock_input")
x_stock = LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001))(stock_input)
x_stock =Dropout(0.2)(x_stock)
x_stock = LSTM(32, kernel_regularizer=l2(0.001))(x_stock)
x_stock =Dropout(0.2)(x_stock)

# NEWS BRANCH
news_input = Input(shape=(TIME_STEP, 4), name="news_input")
x_news = Dense(32, activation="relu")(news_input)
attention_layer = Attention()
news_context = attention_layer([x_news, x_news])
news_vector = LSTM(16)(news_context)
news_vector =Dropout(0.2)(news_vector)

# MERGE stock + news first, then apply Dense
stock_news = Concatenate()([x_stock, news_vector])
stock_news = Dense(32, activation="relu")(stock_news)

# SOX BRANCH
sox_input = Input(shape=(TIME_STEP, 1), name="sox_input")
x_sox = LSTM(16)(sox_input)
x_sox =Dropout(0.2)(x_sox)

# MERGE stock_news + sox
merged = Concatenate()([stock_news, x_sox])
merged = Dense(32, activation="relu")(merged)
merged = Dense(16, activation="relu")(merged)

output = Dense(FORECAST_HORIZON)(merged)

model = Model(inputs=[stock_input, news_input, sox_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="mse"
)
model.summary()
plot_model(
    model,
    to_file="architecture_v1a.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    dpi=192
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)
# ------------------------------
# Training
# ------------------------------
history = model.fit(
    [X_stock[:train_size], X_news[:train_size], X_sox[:train_size]],
    y[:train_size],
    validation_data=(
    [X_stock[train_size + TIME_STEP:], X_news[train_size + TIME_STEP:], X_sox[train_size + TIME_STEP:]],
    y[train_size + TIME_STEP:]
    ),
    epochs=500,
    batch_size=32,
    callbacks=[early_stop]
)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ------------------------------
# Forecast
# ------------------------------
past = 30

last_stock = stock_data.iloc[-TIME_STEP-past:-past]
last_news = news_data.iloc[-TIME_STEP-past:-past]

last_stock_scaled = stock_scaler.transform(last_stock)
last_news_scaled = news_scaler.transform(last_news)

last_stock_scaled = last_stock_scaled.reshape(1, TIME_STEP, 6)
last_news_scaled = last_news_scaled.reshape(1, last_news_scaled.shape[0], 4)
last_sox_scaled = sox_scaler.transform(sox_data.iloc[-TIME_STEP-past:-past].values.reshape(-1, 1))
last_sox_scaled = last_sox_scaled.reshape(1, TIME_STEP, 1)

pred_scaled = model.predict([last_stock_scaled, last_news_scaled, last_sox_scaled])[0]

predicted_close = close_scaler.inverse_transform(
    pred_scaled.reshape(-1,1)
).flatten()

print("\nNext 3 day closing prices:")

for i, price in enumerate(predicted_close, 1):
    print(f"Day +{i}: {price:.2f}")
print(stock_data[["Adj Close"]].iloc[-past:-past+3])
actual_close = stock_data["Adj Close"].iloc[-past:-past+3].values
loss = np.mean(np.abs(predicted_close - actual_close) / actual_close)
print(f"\nMean Absolute Percentage Error: {loss:.2%}")
for i in range(3):
    day_loss = np.abs(predicted_close[i] - actual_close[i]) / actual_close[i]
    print(f"Day +{i+1} MAPE: {day_loss:.2%}")
