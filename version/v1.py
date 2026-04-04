import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

TIME_STEP = 40
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

#------------------------------
# Load PHLX Semiconductor Sector (SOX) index data
#------------------------------
sox_data = pd.read_csv("stock_^SOX_2021-01-01_to_2026-03-07.csv", parse_dates=["Date"], index_col="Date")
sox_data = sox_data[["Close"]]

# ------------------------------
# Align news to stock days
# ------------------------------
#Don't use this if we want to capture news impact on non-market days, since reindexing will drop non-market days and fill them with 0 which may not be accurate. Instead, we will get news between the actual time of stock in the create_dataset function.
news_data = news_data.reindex(stock_data.index).fillna(0)

# ------------------------------
# Scaling
# ------------------------------

stock_scaler = MinMaxScaler()
scaled_stock = stock_scaler.fit_transform(stock_data)

news_scaler = MinMaxScaler()
scaled_news = news_scaler.fit_transform(news_data)

sox_scaler = MinMaxScaler()
scaled_sox = sox_scaler.fit_transform(sox_data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(stock_data[["Close"]])

scaled_stock_df = pd.DataFrame(scaled_stock, index=stock_data.index, columns=features)
scaled_news_df = pd.DataFrame(scaled_news, index=news_data.index, columns=news_features)
scaled_sox_df = pd.DataFrame(scaled_sox, index=sox_data.index, columns=["Close"])
scaled_close_df = pd.DataFrame(scaled_close, index=stock_data.index, columns=["Close"])

# ------------------------------
# Dataset creation
# ------------------------------

def create_dataset(stock_df, news_df, sox_df, close_df):

    X_stock, X_news, X_sox, y = [], [], [], []

    for i in range(len(stock_df) - TIME_STEP - FORECAST_HORIZON + 1):

        X_stock.append(stock_df.iloc[i:i+TIME_STEP].values)
        X_news.append(news_df.iloc[i:i+TIME_STEP].values)
        #make news between the actual time of stock instead of only market days, so we can capture news impact on non-market days
        #get day between first and last day of x_stock
        # start_date = stock_df.index[i]
        # end_date = stock_df.index[i+TIME_STEP-1]
        # X_news.append(news_df.loc[start_date:end_date].values)
        
        X_sox.append(sox_df.iloc[i:i+TIME_STEP].values)
        target = close_df.iloc[i+TIME_STEP:i+TIME_STEP+FORECAST_HORIZON]
        y.append(target.values.flatten())

    # X_news = pad_sequences(
    #         X_news,
    #         padding="post",
    #         value = 0.0,
    #         dtype="float32"
    #     )
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
stock_input = Input(shape=(TIME_STEP, 6))

x_stock = LSTM(64, return_sequences=True)(stock_input)
x_stock = LSTM(64)(x_stock)

#concat sox right into stock branch
sox_input = Input(shape=(TIME_STEP, 1))
x_sox = LSTM(32)(sox_input)
x_stock = Concatenate()([x_stock, x_sox])

# NEWS BRANCH
#news_input = Input(shape=(None, 4))
news_input = Input(shape=(TIME_STEP, 4)) 
#if we want to only capture news on market days, but this may miss news impact on non-market days
#drop filler from padding, since news data is more sparse than stock data, we can use masking to ignore the padded values in the attention layer
#masked_news = Masking(mask_value=0.0)(news_input)

x_news = Dense(64, activation="relu")(news_input)

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

model = Model(inputs=[stock_input, sox_input, news_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="mse"
)
model.summary()
plot_model(
    model,
    to_file="architecture.png",
    show_shapes=True,      # Show tensor shapes
    show_dtype=True,       # Show data types
    show_layer_names=True, # Show layer names
    rankdir="TB",          # Top-to-bottom layout
    dpi=192                # Image resolution
)
# ------------------------------
# Training
# ------------------------------
past = model.fit(
    [X_stock, X_sox, X_news],
    y,
    epochs=150,
    batch_size=32,
    validation_split=0.3
)
plt.plot(past.history["loss"], label="Train Loss")
plt.plot(past.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ------------------------------
# Forecast
# ------------------------------
past = 30

last_stock = stock_data.iloc[-TIME_STEP-past:-past] #from 1/29 to 2/24

last_news = news_data.iloc[-TIME_STEP-past:-past]
#get news between the actual time of stock instead of only market days, so we can capture news impact on non-market days
# start_date = stock_data.index[-TIME_STEP-past]
# end_date = stock_data.index[-past-1]
# last_news = news_data.loc[start_date:end_date]

last_stock_scaled = stock_scaler.transform(last_stock)
last_news_scaled = news_scaler.transform(last_news)

last_stock_scaled = last_stock_scaled.reshape(1, TIME_STEP, 6)
last_news_scaled = last_news_scaled.reshape(1, last_news_scaled.shape[0], 4)
last_sox_scaled = sox_scaler.transform(sox_data.iloc[-TIME_STEP-past:-past].values.reshape(-1, 1))
last_sox_scaled = last_sox_scaled.reshape(1, TIME_STEP, 1)

pred_scaled = model.predict([last_stock_scaled, last_sox_scaled, last_news_scaled])[0]

pred_scaled = np.clip(pred_scaled, 0, 1)

predicted_close = close_scaler.inverse_transform(
    pred_scaled.reshape(-1,1)
).flatten()

print("\nNext 3 day closing prices:")

for i, price in enumerate(predicted_close, 1):
    print(f"Day +{i}: {price:.2f}")
print(stock_data[["Close"]].iloc[-past:-past+3]) #actual price 
#calculate loss = (predicted - actual )/ actual
actual_close = stock_data["Close"].iloc[-past:-past+3].values
loss = np.mean(np.abs(predicted_close - actual_close) / actual_close)
print(f"\nMean Absolute Percentage Error: {loss:.2%}")
#individual day errors
for i in range(3):
    day_loss = np.abs(predicted_close[i] - actual_close[i]) / actual_close[i]
    print(f"Day +{i+1} MAPE: {day_loss:.2%}")