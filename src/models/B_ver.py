import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.utils import plot_model
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler


#settings
TIME_STEP = 40
FORECAST_HORIZON = 3
RNG_SEED = 42
TRAIN_BEFORE_DATE = "2026-02-24"
FORECAST_START_DATE = "2025-05-29"
RUNFORECAST = False



#------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
# ------------------------------
# Load data
# ------------------------------
stock_data = pd.read_csv("././data/raw/stock.csv", parse_dates=["Date"], index_col="Date")
features = ["Adj Close","Close","High","Low","Open","Volume"]
#features = ["Close","High","Low","Open"]
stock_data = stock_data[features]

news_data = pd.read_csv("././data/processed/daily_features.csv", parse_dates=["date"], index_col="date")
news_features = ["net_impact","weighted_horizon","news_count","divergence"]
news_data = news_data[news_features]

sox_data = pd.read_csv("././data/raw/stock_^SOX_2021-01-01_to_2026-03-07.csv", parse_dates=["Date"], index_col="Date")
sox_data = sox_data[["Close"]]
#dump stock data after ####
original_stock_data = deepcopy(stock_data)

stock_data = stock_data[stock_data.index < TRAIN_BEFORE_DATE]
# Align news to stock days
news_data = news_data.reindex(stock_data.index).fillna(0)
sox_data = sox_data.reindex(stock_data.index).ffill()

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

# ------------------------------
# Train/Val/Test split (TIME-BASED)
# ------------------------------

split_idx = int(len(stock_data) * 0.7)
val_idx   = int(len(stock_data) * 0.85)

train_stock = stock_data.iloc[:split_idx]
val_stock   = stock_data.iloc[split_idx:val_idx]
test_stock  = stock_data.iloc[val_idx:]

train_news = news_data.iloc[:split_idx]
val_news   = news_data.iloc[split_idx:val_idx]
test_news  = news_data.iloc[val_idx:]

train_sox = sox_data.iloc[:split_idx]
val_sox   = sox_data.iloc[split_idx:val_idx]
test_sox  = sox_data.iloc[val_idx:]

# ------------------------------
# Scaling (FIT ONLY ON TRAIN)
# ------------------------------
stock_scaler = MinMaxScaler()
news_scaler  = MinMaxScaler()
sox_scaler   = MinMaxScaler()
close_scaler = MinMaxScaler()

stock_scaler.fit(train_stock)
news_scaler.fit(train_news)
sox_scaler.fit(train_sox)
close_scaler.fit(train_stock[["Close"]])

scaled_stock = stock_scaler.transform(stock_data)
scaled_news  = news_scaler.transform(news_data)
scaled_sox   = sox_scaler.transform(sox_data)
scaled_close = close_scaler.transform(stock_data[["Close"]])

scaled_stock_df = pd.DataFrame(scaled_stock, index=stock_data.index, columns=features)
scaled_news_df  = pd.DataFrame(scaled_news,  index=news_data.index,  columns=news_features)
scaled_sox_df   = pd.DataFrame(scaled_sox,   index=sox_data.index,   columns=["Close"])
scaled_close_df = pd.DataFrame(scaled_close, index=stock_data.index, columns=["Close"])

# ------------------------------
# Create sequences
# ------------------------------
X_stock, X_news, X_sox, y = create_dataset(
    scaled_stock_df,
    scaled_news_df,
    scaled_sox_df,
    scaled_close_df
)

# Split sequences
n = len(X_stock)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

X_stock_train, X_stock_val, X_stock_test = X_stock[:train_end], X_stock[train_end:val_end], X_stock[val_end:]
X_news_train,  X_news_val,  X_news_test  = X_news[:train_end],  X_news[train_end:val_end],  X_news[val_end:]
X_sox_train,   X_sox_val,   X_sox_test   = X_sox[:train_end],   X_sox[train_end:val_end],   X_sox[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

# Merge features (KEY SIMPLIFICATION)
X_train = np.concatenate([X_stock_train, X_sox_train, X_news_train], axis=-1)
X_val   = np.concatenate([X_stock_val,   X_sox_val,   X_news_val],   axis=-1)
X_test  = np.concatenate([X_stock_test,  X_sox_test,  X_news_test],  axis=-1)

# ------------------------------
# Model
# ------------------------------
set_seeds(RNG_SEED)
input_layer = Input(shape=(TIME_STEP, X_train.shape[2]))

x = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(input_layer)
x = Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)

#x = Dense(32, activation="relu", kernel_regularizer=l1(1e-4))(x)
x = Dropout(0.2)(x)

x = Dense(16, activation="relu")(x)

output = Dense(FORECAST_HORIZON)(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer="adam", loss="MSE")

model.summary()
plot_model(
    model,
    to_file="v1b.png",
    show_shapes=True,      # Show tensor shapes
    show_dtype=True,       # Show data types
    show_layer_names=True, # Show layer names
    rankdir="TB",          # Top-to-bottom layout
    dpi=192                # Image resolution
)

# ------------------------------
# Training
# ------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()

# ------------------------------
# Test evaluation
# ------------------------------

predictions = model.predict(X_test)

pred_prices = close_scaler.inverse_transform(
    predictions.reshape(-1, 1)
).reshape(predictions.shape)
actual_prices = close_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)
percentage_error = (
    np.abs(pred_prices - actual_prices)
    / actual_prices
) * 100
for day in range(FORECAST_HORIZON):
        day_error = np.mean(percentage_error[:, day])
        print(f"Mean % Error Day {day + 1}: {day_error:.4f}%")
percentage_error_flat = percentage_error.flatten()
plt.hist(
    percentage_error_flat,
    bins=30
)
plt.xlabel("Percentage Error (%)")
plt.ylabel("Frequency")
plt.title("Test Set Prediction Percentage Error Distribution")
plt.show()
print("Mean % Error:", np.mean(percentage_error_flat))
print("Median % Error:", np.median(percentage_error_flat))
print("Max % Error:", np.max(percentage_error_flat))

# ------------------------------
# Forecast
# ------------------------------

if RUNFORECAST:
    future_stock = original_stock_data[
        original_stock_data.index >= FORECAST_START_DATE
    ][features]

    future_news = news_data.reindex(future_stock.index).ffill()

    future_sox = pd.read_csv(
        "././data/raw/stock_^SOX_2021-01-01_to_2026-03-07.csv",
        parse_dates=["Date"],
        index_col="Date"
    )[["Close"]]

    future_sox = future_sox.reindex(future_stock.index).ffill()

    scaled_future_stock = pd.DataFrame(
        stock_scaler.transform(future_stock),
        index=future_stock.index,
        columns=features
    )

    scaled_future_news = pd.DataFrame(
        news_scaler.transform(future_news),
        index=future_news.index,
        columns=news_features
    )

    scaled_future_sox = pd.DataFrame(
        sox_scaler.transform(future_sox),
        index=future_sox.index,
        columns=["Close"]
    )

    scaled_future_close = pd.DataFrame(
        close_scaler.transform(future_stock[["Close"]]),
        index=future_stock.index,
        columns=["Close"]
    )

    Xf_stock, Xf_news, Xf_sox, yf = create_dataset(
        scaled_future_stock,
        scaled_future_news,
        scaled_future_sox,
        scaled_future_close
    )

    forecastx = np.concatenate(
        [Xf_stock, Xf_sox, Xf_news],
        axis=-1
    )


    # ------------------------------
    # Predict
    # ------------------------------
    predictions = model.predict(forecastx)
    # inverse transform
    pred_prices = close_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).reshape(predictions.shape)

    actual_prices = close_scaler.inverse_transform(
        yf.reshape(-1, 1)
    ).reshape(yf.shape)

    # ------------------------------
    # Percentage error
    # ------------------------------
    percentage_error = (
        np.abs(pred_prices - actual_prices)
        / actual_prices
    ) * 100
    for day in range(FORECAST_HORIZON):
            day_error = np.mean(percentage_error[:, day])
            print(f"Mean % Error Day {day + 1}: {day_error:.4f}%")
    # flatten all horizons into 1D
    percentage_error_flat = percentage_error.flatten()

    # ------------------------------
    # Histogram
    # ------------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        percentage_error_flat,
        bins=30
    )

    plt.xlabel("Percentage Error (%)")
    plt.ylabel("Frequency")
    plt.title("Prediction Percentage Error Distribution")

    plt.show()

    # ------------------------------
    # Statistics
    # ------------------------------
    
    print("Mean % Error:", np.mean(percentage_error_flat))
    print("Median % Error:", np.median(percentage_error_flat))
    print("Max % Error:", np.max(percentage_error_flat))