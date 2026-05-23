from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output single value for predicted price

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    return model

