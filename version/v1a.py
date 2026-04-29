import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention
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
# Scaling  (train 70% / val 15% / test 15%)
# ------------------------------
# 先在「原始資料」上找兩個切點：70% 給 train，70%~85% 給 val，85%~100% 給 test
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

# STOCK BRANCH
stock_input = Input(shape=(TIME_STEP, 6), name="stock_input")
x_stock = LSTM(32, return_sequences=True)(stock_input)

x_stock = LSTM(32)(x_stock)

# NEWS BRANCH
news_input = Input(shape=(TIME_STEP, 4), name="news_input")
x_news = Dense(32, activation="relu")(news_input)
attention_layer = Attention()
news_context = attention_layer([x_news, x_news])
news_vector = LSTM(16)(news_context)


# MERGE stock + news first, then apply Dense
stock_news = Concatenate()([x_stock, news_vector])
stock_news = Dense(32, activation="relu")(stock_news)

# SOX BRANCH
sox_input = Input(shape=(TIME_STEP, 1), name="sox_input")
x_sox = LSTM(32)(sox_input)


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

# ------------------------------
# Training
# ------------------------------
# 切出三段 sequence（train / val / test），每段邊界都留 TIME_STEP 緩衝避免洩漏
X_stock_train, X_news_train, X_sox_train = (
    X_stock[:train_size], X_news[:train_size], X_sox[:train_size]
)
y_train = y[:train_size]

X_stock_val, X_news_val, X_sox_val = (
    X_stock[train_size + TIME_STEP : val_size],
    X_news[train_size + TIME_STEP : val_size],
    X_sox[train_size + TIME_STEP : val_size],
)
y_val = y[train_size + TIME_STEP : val_size]

X_stock_test, X_news_test, X_sox_test = (
    X_stock[val_size + TIME_STEP :],
    X_news[val_size + TIME_STEP :],
    X_sox[val_size + TIME_STEP :],
)
y_test = y[val_size + TIME_STEP :]

print(f"Train sequences: {len(y_train)}, Val sequences: {len(y_val)}, Test sequences: {len(y_test)}")

history = model.fit(
    [X_stock_train, X_news_train, X_sox_train],
    y_train,
    validation_data=(
        [X_stock_val, X_news_val, X_sox_val],
        y_val,
    ),
   epochs=150,
    batch_size=32,
)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ------------------------------
# Test evaluation：完整拆解 + 對照 baseline + 散佈圖
# ------------------------------
print("\n" + "=" * 60)
print("Test set evaluation (predicting daily returns)")
print("=" * 60)

test_loss = model.evaluate(
    [X_stock_test, X_news_test, X_sox_test],
    y_test,
    verbose=0,
)
print(f"Test Loss (MSE on returns, all 3 days aggregated): {test_loss:.6f}")

# 拿到 test 上的預測（仍是 return 尺度）
pred_test = model.predict(
    [X_stock_test, X_news_test, X_sox_test], verbose=0
)


def report(actual, pred, label):
    """印 MSE / MAE / RMSE，return 尺度。"""
    mse = np.mean((actual - pred) ** 2)
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(mse)
    print(f"  {label:32s}  MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}")
    return mse


# Baseline: 永遠預測 0 報酬率（≡ 隔天股價 = 今天股價，naive）
baseline_zero = np.zeros_like(y_test)

print("\n--- Aggregated over all 3 days ---")
mse_model = report(y_test, pred_test,    "Model (LSTM)")
mse_naive = report(y_test, baseline_zero,"Baseline: predict 0 return")
print(f"  Improvement over naive baseline: "
      f"{(1 - mse_model / mse_naive) * 100:+.2f}%")

print("\n--- Per-day breakdown ---")
for d in range(FORECAST_HORIZON):
    print(f"  Day +{d+1}:")
    report(y_test[:, d], pred_test[:, d],    "    Model")
    report(y_test[:, d], baseline_zero[:, d],"    Baseline (zero)")

# 預測 / 實際的分布診斷
print("\n--- Distribution diagnostics (return scale) ---")
print(f"  Pred  : mean={pred_test.mean():+.5f}  std={pred_test.std():.5f}  "
      f"min={pred_test.min():+.5f}  max={pred_test.max():+.5f}")
print(f"  Actual: mean={y_test.mean():+.5f}  std={y_test.std():.5f}  "
      f"min={y_test.min():+.5f}  max={y_test.max():+.5f}")

# ------------------------------
# Scatter plot: predicted vs actual (各 day 一張)
# ------------------------------
fig, axes = plt.subplots(1, FORECAST_HORIZON, figsize=(14, 4.5))
for d in range(FORECAST_HORIZON):
    ax = axes[d]
    actual_pct = y_test[:, d] * 100
    pred_pct   = pred_test[:, d] * 100

    ax.scatter(actual_pct, pred_pct, alpha=0.5, s=18, edgecolor="none")

    # y = x 參考線
    lim = max(np.abs(actual_pct).max(), np.abs(pred_pct).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, label="y = x (perfect)")

    # 0 軸
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # 簡單線性擬合，呈現「預測 vs 實際」的傾斜趨勢
    if actual_pct.std() > 0:
        slope, intercept = np.polyfit(actual_pct, pred_pct, 1)
        xs = np.linspace(-lim, lim, 100)
        ax.plot(xs, slope * xs + intercept, "b-", linewidth=1.2,
                label=f"fit: slope={slope:.3f}")

    ax.set_xlabel("Actual return (%)")
    ax.set_ylabel("Predicted return (%)")
    ax.set_title(f"Day +{d+1}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left", fontsize=8)

plt.suptitle("v1a — Predicted vs Actual daily returns (Test set)")
plt.tight_layout()
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

pred_returns = model.predict([last_stock_scaled, last_news_scaled, last_sox_scaled])[0]

# 用預測報酬率還原成絕對價格
last_known_price = stock_data["Adj Close"].iloc[-past-1]
predicted_close = []
price = last_known_price
for r in pred_returns:
    price = price * (1 + r)
    predicted_close.append(price)
predicted_close = np.array(predicted_close)


pred_scaled_diag = model.predict([X_stock[:50], X_news[:50], X_sox[:50]])
pred_orig = pred_scaled_diag
y_orig = y[:50]
print("Predicted returns (original scale):", pred_orig[:5])
print("Actual returns (original scale):", y_orig[:5])
print("Pred std:", pred_orig.std(), "Actual std:", y_orig.std())
actual_close = stock_data["Adj Close"].iloc[-past:-past+3].values
actual_dates = stock_data.index[-past:-past+3]
actual_prev_prices = np.concatenate([[last_known_price], actual_close[:-1]])
actual_pct_changes = (actual_close - actual_prev_prices) / actual_prev_prices * 100

print("\nNext 3 day closing prices:")
print(f"{'':10} {'Predicted':>12} {'Actual':>12}")
print("-" * 40)
for i in range(FORECAST_HORIZON):
    pred_pct = pred_returns[i] * 100
    pred_sign = "+" if pred_pct >= 0 else ""
    act_pct = actual_pct_changes[i]
    act_sign = "+" if act_pct >= 0 else ""
    date_str = actual_dates[i].strftime("%Y-%m-%d")
    print(
        f"{date_str}  "
        f"{predicted_close[i]:>7.2f} ({pred_sign}{pred_pct:.2f}%)  "
        f"{actual_close[i]:>7.2f} ({act_sign}{act_pct:.2f}%)"
    )

loss = np.mean(np.abs(predicted_close - actual_close) / actual_close)
print(f"\nMean Absolute Percentage Error: {loss:.2%}")
for i in range(3):
    day_loss = np.abs(predicted_close[i] - actual_close[i]) / actual_close[i]
    print(f"Day +{i+1} MAPE: {day_loss:.2%}")
