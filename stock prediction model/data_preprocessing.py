import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def fetch_data(stock_symbol, start_date='2015-01-01', end_date='2023-01-01'):
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    # Check if dataframe is empty
    if df.empty:
        print("No data found. Check stock symbol or internet connection.")
        return None

    return df


def preprocess_data(df):

    # Check again before processing
    if df is None or df.empty:
        raise ValueError("Dataframe is empty.")

    df = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

    x_data, y_data = [], []

    for i in range(60, len(scaled_data)):
        x_data.append(scaled_data[i-60:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data, y_data = np.array(x_data), np.array(y_data)

    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, shuffle=False
    )

    return x_train, x_test, y_train, y_test, scaler