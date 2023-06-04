import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

data = pd.read_csv('HDFCBANK.csv')
prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)


train_size = int(len(scaled_prices) * 0.9)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]


n_steps = 10 
X_train, y_train = [], []
for i in range(n_steps, len(train_data)):
    X_train.append(train_data[i - n_steps:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=10)


X_test, y_test = [], []
for i in range(n_steps, len(test_data)):
    X_test.append(test_data[i - n_steps:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predictions = model.predict(X_test)


predictions = scaler.inverse_transform(predictions)


rmse = np.mean((np.abs((predictions - scaler.inverse_transform(y_test.reshape(-1, 1)))))/(scaler.inverse_transform(y_test.reshape(-1, 1))))*100
print("Percentage error:", rmse)
