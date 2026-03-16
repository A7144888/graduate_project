import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate,Softmax,Dot, Reshape
from sklearn.preprocessing import MinMaxScaler

_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
_DATA_RAW = os.path.join(_BASE, "data", "raw")
_DATA_PROCESSED = os.path.join(_BASE, "data", "processed")
_DATA_OUTPUT = os.path.join(_BASE, "data", "output")

#------------------------------
#global variables
#------------------------------
time_step = 20
forecast_horizon = 3
#------------------------------
# model section
#------------------------------

#left line, process stock prices, use lstm
stock_input = Input(shape=(time_step, 6))
stock_lstm = LSTM(50, return_sequences=True)(stock_input)
stock_lstm = LSTM(50)(stock_lstm)
#right line, process news sentiment, use a dense and an attention layer

news_input = Input(shape=(None, 4))
news_dense = Dense(50, activation="relu")(news_input)
attention = Dense(1, activation="tanh")(news_dense)
attention = Softmax(axis=1)(attention)
news_vector = Dot(axes=1)([attention, news_dense])
news_vector = Reshape((50,))(news_vector)
#merge the two lines, use a dense layer and an attention layer to output the final prediction
merged = Concatenate()([stock_lstm, news_vector])
merged = Dense(50, activation="relu")(merged)
merged = Dense(32, activation="relu")(merged)
output = Dense(forecast_horizon)(merged)

#------------------------------
#data section
#------------------------------

#get continuous stock price data of 20 trading days, and preprocess it
stock_data = pd.read_csv(os.path.join(_DATA_RAW, "stock.csv"), parse_dates=["Date"], index_col="Date")
features = ["Adj Close","Close","High","Low","Open","Volume"]
stock_data = stock_data[features]
scaler = MinMaxScaler()
scaled_stock_data = scaler.fit_transform(stock_data.values)
#get continuous news sentiment data of the first trading day to the last trading day of x
news_data = pd.read_csv(os.path.join(_DATA_PROCESSED, "daily_features.csv"), parse_dates=["date"], index_col="date")
news_features = ["net_impact","weighted_horizon","news_count","divergence"]
news_data = news_data[news_features]
news_scaler = MinMaxScaler()
scaled_news_data = news_scaler.fit_transform(news_data.values)

close_scaler = MinMaxScaler()
close_prices = stock_data[["Close"]]
scaled_close = close_scaler.fit_transform(close_prices)

def create_dataset(stock_df, news_df,close_df):
    X_stock = []
    X_news = []
    y = []

    for i in range(len(stock_df) - time_step - forecast_horizon + 1):
        stock_window = stock_df.iloc[i : i + time_step]
        start_date = stock_window.index[0]
        end_date = stock_window.index[-1]

        # Get news for this window
        news_window = news_df.loc[start_date:end_date]

        X_stock.append(stock_window.values)
        X_news.append(news_window.values)
        # Target: Next 3 days of "Close"
        y.append(close_df.iloc[i + time_step : i + time_step + forecast_horizon].values)

    # Convert stock to 3D array: (Samples, Time_Steps, Features)
    X_stock = np.array(X_stock)
    
    # Pad news to the same length: (Samples, Max_News_Length, Features)
    # This fixes the "different sizes" issue in your error log
    X_news = pad_sequences(X_news, padding="post", dtype="float32")
    
    y = np.array(y)

    return X_stock, X_news, y

# Generate data
xstck, xnws, y = create_dataset(pd.DataFrame(scaled_stock_data, index=stock_data.index,columns=features), 
                                pd.DataFrame(scaled_news_data, index=news_data.index,columns=news_features),
                                pd.DataFrame(scaled_close, index=stock_data.index,columns=["Close"]))


#------------------------------
#training section
#------------------------------

model = Model(inputs=[stock_input, news_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="mse"
)
model.fit([xstck, xnws], y, epochs=30, batch_size=32)
model.summary()

#------------------------------
#forecasting section
#------------------------------

# get last stock window
last_stock_window = stock_data.iloc[-time_step:]

start_date = last_stock_window.index[0]
end_date = last_stock_window.index[-1]

# get all news within this stock window
last_news_window = news_data.loc[start_date:end_date]

# scale them
last_stock_scaled = scaler.transform(last_stock_window.values)
last_news_scaled = news_scaler.transform(last_news_window.values)

# reshape stock for model
last_stock_scaled = last_stock_scaled.reshape(1, time_step, 6)

# pad news sequence
last_news_scaled = pad_sequences(
    [last_news_scaled],
    padding="post",
    dtype="float32"
)

# predict
pred_scaled = model.predict([last_stock_scaled, last_news_scaled])

# ------------------------------
# inverse scaling (Close only)
# ------------------------------

close_index = features.index("Close")

dummy = np.zeros((forecast_horizon, len(features)))
dummy[:, close_index] = pred_scaled[0]

predicted_close = scaler.inverse_transform(dummy)[:, close_index]

print("Next 3 day closing prices:")
for i, price in enumerate(predicted_close, 1):
    print(f"Day +{i}: {price:.2f}")

last_date = stock_data.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=forecast_horizon + 1)[1:]

result_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Price": predicted_close
})
result_df.to_csv(os.path.join(_DATA_OUTPUT, "LSTMAttention.csv"), index=False)
print("Results saved to LSTMAttention.csv")
