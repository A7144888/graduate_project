import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler

TIME_STEP = 40
FORECAST_HORIZON = 3

# ------------------------------
# Load data
# ------------------------------
stock_data = pd.read_csv("stock.csv", parse_dates=["Date"], index_col="Date")
features = ["Adj Close","Close","High","Low","Open","Volume"]
stock_data = stock_data[features]

news_data = pd.read_csv("daily_features.csv", parse_dates=["date"], index_col="date")
news_features = ["net_impact","weighted_horizon","news_count","divergence"]
news_data = news_data[news_features]

sox_data = pd.read_csv("stock_^SOX_2021-01-01_to_2026-03-07.csv", parse_dates=["Date"], index_col="Date")
sox_data = sox_data[["Close"]]

# Align news to stock days
news_data = news_data.reindex(stock_data.index).fillna(0)
sox_data = sox_data.reindex(stock_data.index).fillna(method="ffill")

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
# Model (SIMPLE + REGULARIZED)
# ------------------------------
input_layer = Input(shape=(TIME_STEP, X_train.shape[2]))

x = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(input_layer)
x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(x)

#x = Dense(32, activation="relu", kernel_regularizer=l1(1e-4))(x)
x = Dropout(0.2)(x)

x = Dense(16, activation="relu")(x)

output = Dense(FORECAST_HORIZON)(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer="adam", loss="mse")

model.summary()
plot_model(
    model,
    to_file="v2.png",
    show_shapes=True,      # Show tensor shapes
    show_dtype=True,       # Show data types
    show_layer_names=True, # Show layer names
    rankdir="TB",          # Top-to-bottom layout
    dpi=192                # Image resolution
)

# ------------------------------
# Training
# ------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

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
test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

# ------------------------------
# Forecast
# ------------------------------
run = False
if(run):
    past = 15

    last_stock = test_stock.iloc[-TIME_STEP-past:-past]
    last_news  = test_news.iloc[-TIME_STEP-past:-past]
    last_sox   = test_sox.iloc[-TIME_STEP-past:-past]

    last_stock_scaled = stock_scaler.transform(last_stock)
    last_news_scaled  = news_scaler.transform(last_news)
    last_sox_scaled   = sox_scaler.transform(last_sox)

    last_combined = np.concatenate(
        [last_stock_scaled, last_sox_scaled, last_news_scaled],
        axis=-1
    ).reshape(1, TIME_STEP, -1)

    pred_scaled = model.predict(last_combined)[0]
    pred_scaled = np.clip(pred_scaled, 0, 1)

    predicted_close = close_scaler.inverse_transform(
        pred_scaled.reshape(-1,1)
    ).flatten()

    print("\nNext 3 day closing prices:")
    for i, price in enumerate(predicted_close, 1):
        print(f"Day +{i}: {price:.2f}")

    # Compare with actual
    actual_close = stock_data["Close"].iloc[-past:-past+3].values

    mape = np.mean(np.abs(predicted_close - actual_close) / actual_close)
    print(f"\nMAPE: {mape:.2%}")

    for i in range(3):
        day_loss = np.abs(predicted_close[i] - actual_close[i]) / actual_close[i]
        print(f"Day +{i+1} MAPE: {day_loss:.2%}")