import matplotlib.pyplot as plt
from data_preprocessing import fetch_data, preprocess_data
from stock_prediction_model import build_lstm_model, train_model
from real_time_prediction import predict_next_day

# Main workflow
if __name__ == '__main__':
    stock_symbol = 'AAPL'  
    
   
    df = fetch_data(stock_symbol)
    
    
    x_train, x_test, y_train, y_test, scaler = preprocess_data(df)
    
 
    model = build_lstm_model((x_train.shape[1], 1))
    
   
    model = train_model(x_train, y_train, x_test, y_test, model)
    
   
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
  
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(actual_prices, color='blue', label='Actual Stock Price')
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  
    plt.show()  
    
  
    recent_data = df['Close'].values[-60:] 
    predicted_price = predict_next_day(model, recent_data, scaler)
    print(f"Predicted stock price for {stock_symbol} tomorrow: ${predicted_price:.2f}")
    
    
    with open('predicted_prices.txt', 'w') as file:
        file.write("Predicted Prices:\n")
        file.write(str(predicted_prices[:100]))  

