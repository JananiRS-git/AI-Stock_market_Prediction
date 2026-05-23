import numpy as np
from tensorflow.keras.models import load_model

def predict_next_day(model, recent_data, scaler):
    recent_data = np.reshape(recent_data, (1, recent_data.shape[0], 1))  # Reshape for LSTM
    predicted_price = model.predict(recent_data)
    predicted_price = scaler.inverse_transform(predicted_price)  # Inverse scaling
    return predicted_price[0][0]

