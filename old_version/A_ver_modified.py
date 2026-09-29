import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention,LayerNormalization,GlobalAveragePooling1D,Multiply

TIME_STEP = 40
FORECAST_HORIZON = 3
RNG_SEED = 42
run = True
TRAIN_BEFORE_DATE = "2024-01-01"
forecast_start_date = "2025-01-01"


#------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



#   BASE_LOSS: "mse" | "rmse" | "mae" | "huber" | "logcosh"
#   VARIANCE_PENALTY_ALPHA: 變異性懲罰權重；0.0 = 純 base loss，越大越強迫 Pred std → Actual std
BASE_LOSS = "rmse"
HUBER_DELTA = 0.005
VARIANCE_PENALTY_ALPHA = 0.8

def _base_loss(y_true, y_pred):
    if BASE_LOSS == "mse":
        return tf.reduce_mean(tf.square(y_true - y_pred))
    if BASE_LOSS == "rmse":
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return tf.sqrt(tf.maximum(mse, tf.keras.backend.epsilon()))
    if BASE_LOSS == "mae":
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    if BASE_LOSS == "huber":
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, HUBER_DELTA)
        linear = abs_err - quadratic
        return tf.reduce_mean(0.5 * tf.square(quadratic) + HUBER_DELTA * linear)
    if BASE_LOSS == "logcosh":
        return tf.reduce_mean(tf.math.log(tf.math.cosh(y_pred - y_true)))
    raise ValueError(f"Unknown BASE_LOSS: {BASE_LOSS}")


def variance_aware_loss(y_true, y_pred):
    """Base loss + 對「預測 std 跟真實 std 的差」做平方懲罰（按 forecast day 各算一次）。"""
    base = _base_loss(y_true, y_pred)
    pred_std = tf.math.reduce_std(y_pred, axis=0)
    target_std = tf.math.reduce_std(y_true, axis=0)
    var_penalty = tf.reduce_mean(tf.square(pred_std - target_std))
    return base + VARIANCE_PENALTY_ALPHA * var_penalty


def resolve_anchor_after_reference_date(
    trading_index: pd.DatetimeIndex,
    time_step: int,
    horizon: int,
    after_date: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """依參考日取得錨點（最後一個已知收盤）與往後 horizon 個預測交易日。"""
    ref = pd.Timestamp(after_date)
    future = trading_index[trading_index > ref]
    if len(future) < horizon:
        raise ValueError(
            f"FORECAST_AFTER_DATE={ref.date()} 之後不足 {horizon} 個交易日（請延長 CSV 或改早參考日）"
        )
    forecast_dates = pd.DatetimeIndex(future[:horizon])
    past = trading_index[trading_index < forecast_dates[0]]
    if len(past) == 0:
        raise ValueError(
            f"在首個預測日 {forecast_dates[0].strftime('%Y-%m-%d')} 之前沒有歷史列"
        )
    anchor_ts = pd.Timestamp(past[-1])
    anchor_pos = trading_index.get_loc(anchor_ts)
    if not isinstance(anchor_pos, (int, np.integer)):
        raise ValueError("股價索引含重複日期，請先整理 Date 欄")
    anchor_pos = int(anchor_pos)
    if anchor_pos < time_step - 1:
        raise ValueError(
            f"錨點 {anchor_ts.strftime('%Y-%m-%d')} 之前不足 {time_step} 天歷史，無法組 LSTM 視窗"
        )
    return anchor_ts, forecast_dates


# ------------------------------
# Load stock data
# ------------------------------

stock_data = pd.read_csv("././data/raw/stock.csv", parse_dates=["Date"], index_col="Date")
features = ["Adj Close","Close","High","Low","Open","Volume"]
stock_data = stock_data[features]
original_stock_data = deepcopy(stock_data)

# ------------------------------
# Load news data
# ------------------------------

news_data = pd.read_csv("././data/processed/daily_features.csv", parse_dates=["date"], index_col="date")

news_features = ["net_impact","weighted_horizon","news_count","divergence"]
news_data = news_data[news_features]
original_news_data = deepcopy(news_data)

#------------------------------
# Load PHLX Semiconductor Sector (SOX) index data
#------------------------------
sox_data = pd.read_csv("././data/raw/stock_^SOX_2021-01-01_to_2026-03-07.csv", parse_dates=["Date"], index_col="Date")
sox_data = sox_data[["Adj Close"]]
original_sox_data = deepcopy(sox_data)
# ------------------------------
# Align news to stock days
# ------------------------------
news_data = news_data.reindex(stock_data.index).fillna(0)
sox_data = sox_data.reindex(stock_data.index).ffill().bfill()
sox_data = sox_data.shift(1).bfill()

_adj_close_full = stock_data["Adj Close"].copy()
_anchor_ts, _forecast_dates = resolve_anchor_after_reference_date(
    stock_data.index,
    TIME_STEP,
    FORECAST_HORIZON,
    TRAIN_BEFORE_DATE
)
stock_data = stock_data.loc[:_anchor_ts]
news_data = news_data.loc[:_anchor_ts]
sox_data = sox_data.loc[:_anchor_ts]

# ------------------------------
# Scaling  (train 70% / val 15% / test 15%)
# ------------------------------
split_train = int(len(stock_data) * 0.7)
split_val   = int(len(stock_data) * 0.85)

# 換算成 sequence 陣列的索引
# 一筆 sequence 用到 [i, i+TIME_STEP+FORECAST_HORIZON) 這段資料
# 對應原始第 N 天前能形成的最後一筆 sequence index = N - TIME_STEP - FORECAST_HORIZON + 1
train_size = split_train - TIME_STEP - FORECAST_HORIZON + 1
val_size   = split_val   - TIME_STEP - FORECAST_HORIZON + 1

train_stock = stock_data.iloc[:split_train]
train_news  = news_data.iloc[:split_train]
train_sox   = sox_data.iloc[:split_train]


stock_scaler = MinMaxScaler()
stock_scaler.fit(train_stock)
scaled_stock = stock_scaler.transform(stock_data)

news_scaler = MinMaxScaler()
news_scaler.fit(train_news)
scaled_news = news_scaler.transform(news_data)

sox_scaler = MinMaxScaler()
sox_scaler.fit(train_sox)
scaled_sox = sox_scaler.transform(sox_data)

scaled_stock_df = pd.DataFrame(scaled_stock, index=stock_data.index, columns=features)
scaled_news_df = pd.DataFrame(scaled_news, index=news_data.index, columns=news_features)
scaled_sox_df = pd.DataFrame(scaled_sox, index=sox_data.index, columns=["Adj Close"])

# 改預測每日報酬率（漲跌幅）而非絕對價格
returns = stock_data["Adj Close"].pct_change().fillna(0)
returns_df = pd.DataFrame(returns.values, index=stock_data.index, columns=["return"])

# ------------------------------
# Dataset creation
# ------------------------------

def create_dataset(stock_df, news_df, sox_df, return_df):

    X_stock, X_news, X_sox, y = [], [], [], []

    for i in range(len(stock_df) - TIME_STEP - FORECAST_HORIZON + 1):

        X_stock.append(stock_df.iloc[i:i+TIME_STEP].values)
        X_news.append(news_df.iloc[i:i+TIME_STEP].values)
        X_sox.append(sox_df.iloc[i:i+TIME_STEP].values)
        target = return_df.iloc[i+TIME_STEP:i+TIME_STEP+FORECAST_HORIZON]
        y.append(target.values.flatten())

    return np.array(X_stock), np.array(X_news), np.array(X_sox), np.array(y)

X_stock, X_news, X_sox, y = create_dataset(
    scaled_stock_df,
    scaled_news_df,
    scaled_sox_df,
    returns_df
)

# ------------------------------
# Model
# ------------------------------
set_seeds(RNG_SEED)

# STOCK BRANCH
stock_input = Input(shape=(TIME_STEP, 6), name="stock_input")
x_stock = LSTM(32, return_sequences=True)(stock_input)
x_stock= LayerNormalization()(x_stock)#
x_stock = LSTM(32, return_sequences=True)(x_stock)
x_stock= LayerNormalization()(x_stock)#
# NEWS BRANCH
news_input = Input(shape=(TIME_STEP, 4), name="news_input")
x_news = LSTM(32)(news_input)
#attention_layer = Attention()
#x_news = attention_layer([x_news, x_news])
news_vector = LayerNormalization()(x_news)

# SOX BRANCH
sox_input = Input(shape=(TIME_STEP, 1), name="sox_input")
x_sox = LSTM(32,return_sequences=True)(sox_input)
x_sox = LayerNormalization()(x_sox)#

# concat sox with stock because of same properties, use attention to concat
attention_layer = Attention()
stock_sox_concat = attention_layer([x_stock, x_sox])
stock_sox_concat = GlobalAveragePooling1D()(stock_sox_concat)
# MERGE stock_news + sox
merged = Multiply()([stock_sox_concat, news_vector])
merged = Dense(32, activation="relu")(merged)
merged = Dense(16, activation="relu")(merged)

output = Dense(FORECAST_HORIZON)(merged)

model = Model(inputs=[stock_input, news_input, sox_input], outputs=output)

model.compile(
    optimizer="adam",
    loss=variance_aware_loss,
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

# ------------------------------
# Training
# ------------------------------
# 切出三段 sequence（train / val / test），每段邊界都留 TIME_STEP 緩衝避免洩漏
X_stock_train, X_news_train, X_sox_train = (
    X_stock[:train_size], X_news[:train_size], X_sox[:train_size]
)
y_train = y[:train_size]

X_stock_val, X_news_val, X_sox_val = (
    X_stock[train_size:val_size],
    X_news[train_size:val_size],
    X_sox[train_size:val_size],
)
y_val = y[train_size:val_size]

X_stock_test, X_news_test, X_sox_test = (
    X_stock[val_size:],
    X_news[val_size:],
    X_sox[val_size:],
)
y_test = y[val_size:]

print(f"Train sequences: {len(y_train)}, Val sequences: {len(y_val)}, Test sequences: {len(y_test)}")

history = model.fit(
    [X_stock_train, X_news_train, X_sox_train],
    y_train,
    validation_data=(
        [X_stock_val, X_news_val, X_sox_val],
        y_val,
    ),
   epochs=100,
    batch_size=32,
)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

X_test = [X_stock_test, X_news_test, X_sox_test]

predictions = model.predict(X_test)


reconstructed_pred_prices = []
reconstructed_actual_prices = []

for i in range(len(predictions)):

    # last known actual price before forecast horizon
    anchor_price = stock_data["Adj Close"].iloc[
        val_size + TIME_STEP + i - 1
    ]

    pred_seq = []
    actual_seq = []

    pred_price = anchor_price
    actual_price = anchor_price

    for h in range(FORECAST_HORIZON):

        pred_price *= (1 + predictions[i, h])
        actual_price *= (1 + y_test[i, h])

        pred_seq.append(pred_price)
        actual_seq.append(actual_price)

    reconstructed_pred_prices.append(pred_seq)
    reconstructed_actual_prices.append(actual_seq)

pred_prices = np.array(reconstructed_pred_prices)
actual_prices = np.array(reconstructed_actual_prices)
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


if run:

    # ---------------------------------
    # Use original FULL datasets
    # ---------------------------------
    future_stock = original_stock_data[
        original_stock_data.index >= forecast_start_date
    ][features]

    future_news = original_news_data.reindex(future_stock.index).ffill()


    future_sox = original_sox_data.reindex(future_stock.index).ffill()

    # ---------------------------------
    # Scaling
    # ---------------------------------
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
        columns=["Adj Close"]
    )

    # ---------------------------------
    # RETURNS TARGET
    # ---------------------------------
    future_returns = (
        future_stock["Adj Close"]
        .pct_change()
        .fillna(0)
    )

    future_returns_df = pd.DataFrame(
        future_returns.values,
        index=future_stock.index,
        columns=["return"]
    )

    # ---------------------------------
    # Create dataset
    # ---------------------------------
    Xf_stock, Xf_news, Xf_sox, yf = create_dataset(
        scaled_future_stock,
        scaled_future_news,
        scaled_future_sox,
        future_returns_df
    )

    # ---------------------------------
    # Predict RETURNS
    # ---------------------------------
    predictions = model.predict(
        [Xf_stock, Xf_news, Xf_sox]
    )


    # ---------------------------------
    # Convert returns -> prices
    # ---------------------------------
    reconstructed_pred_prices = []
    reconstructed_actual_prices = []

    for i in range(len(predictions)):

        # last known price before prediction window
        anchor_price = future_stock["Adj Close"].iloc[
            i + TIME_STEP - 1
        ]

        pred_seq = []
        actual_seq = []

        pred_price = anchor_price
        actual_price = anchor_price

        for h in range(FORECAST_HORIZON):

            pred_price *= (1 + predictions[i, h])
            actual_price *= (1 + yf[i, h])

            pred_seq.append(pred_price)
            actual_seq.append(actual_price)

        reconstructed_pred_prices.append(pred_seq)
        reconstructed_actual_prices.append(actual_seq)

    pred_prices = np.array(reconstructed_pred_prices)
    actual_prices = np.array(reconstructed_actual_prices)

    # ---------------------------------
    # Percentage Error
    # ---------------------------------
    percentage_error = (
        np.abs(pred_prices - actual_prices)
        / actual_prices
    ) * 100
    for day in range(FORECAST_HORIZON):
        day_error = np.mean(percentage_error[:, day])
        print(f"Mean % Error Day {day + 1}: {day_error:.4f}%")
    percentage_error_flat = percentage_error.flatten()

    # ---------------------------------
    # Histogram
    # ---------------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        percentage_error_flat,
        bins=30
    )

    plt.xlabel("Percentage Error (%)")
    plt.ylabel("Frequency")
    plt.title("Forecast Percentage Error Distribution")

    plt.show()

    # ---------------------------------
    # Statistics
    # ---------------------------------
    print("Mean % Error:", np.mean(percentage_error_flat))
    print("Median % Error:", np.median(percentage_error_flat))
    print("Max % Error:", np.max(percentage_error_flat))