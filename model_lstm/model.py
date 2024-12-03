
import os
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanSquaredError

from sklearn.pipeline import Pipeline



def load_and_process_data(first_date, url='http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'):
    df = pd.read_html(url, attrs={'class': 'dxgvTable'}, encoding="UTF-8",thousands='.  ', decimal=',', parse_dates=True, header=0 )[0]
    df = df.rename(columns={'Pre\u00e7o - petr\u00f3leo bruto - Brent (FOB)': 'Close'})
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    df = df[df['Data'] > first_date]
    df.reset_index(drop=True, inplace=True)
    df.sort_index(ascending=False, inplace=True)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['Close']])
    return df, data_scaled, scaler

def create_sequences(data, loopback):
    X, y = [], []
    for i in range(len(data) - loopback):
        X.append(data[i:i+loopback])
        y.append(data[i+loopback])
    return np.array(X), np.array(y)
def build_lstm_model(loopback):
    model = Sequential()
    model.add(LSTM(300, activation='relu', input_shape=(loopback,1)))
    model.add(Dense(1)),

    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model

def evaluate_lstm_model(model, X_test, y_test, scaler):
    lstm_predictions = model.predict(X_test)
    r2_lstm = r2_score(y_test, lstm_predictions)
    mse_lstm = mean_squared_error(y_test, lstm_predictions)
    mae_lstm = mean_absolute_error(y_test, lstm_predictions)
    
    rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    return r2_lstm, mse_lstm, mae_lstm, rmse_lstm

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler file not found.")
    model_lstm = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model_lstm, scaler

def predict(num_prediction, data_scaled, sequence_length, model_lstm, scaler):
    try:
        
        if len(data_scaled) < sequence_length:
            raise ValueError("O comprimento dos dados escalados é menor que o comprimento da sequência.")

        prediction_list = list(data_scaled[-sequence_length:]) 
        print(f"Tamanho inicial da lista de previsões: {len(prediction_list)}")

        for _ in range(num_prediction):
            x = np.array(prediction_list[-sequence_length:]).reshape((1, sequence_length, 1))
            out = model_lstm.predict(x)[0][0]
            print(f"Previsão gerada: {out}")
            prediction_list.append([out]) 

        prediction_list = prediction_list[-num_prediction:]
        print(f"Tamanho da lista de previsões após adicionar novas previsões: {len(prediction_list)}")

        prediction_list = scaler.inverse_transform(np.array(prediction_list).reshape(-1, 1))
        print(f"Previsões desnormalizadas: {prediction_list}")

        return prediction_list

    except FileNotFoundError as fnfe:
        print(f"Erro ao carregar o modelo ou escalador: {fnfe}")
    except ValueError as ve:
        print(f"Erro no valor dos dados: {ve}")
    except Exception as e:
        print(f"Erro durante a previsão: {e}")
    return None

def predict_dates(num_prediction, data):
    last_date = data['Data'].iloc[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates[1:]

def train_and_save_lstm_model(model_path, scaler_path, first_date, sequence_length):
    pipeline = Pipeline([
        ('lstm', build_lstm_model(sequence_length))
    ])
    df, data_scaled, scaler = load_and_process_data(first_date)
    train_size = int(len(data_scaled) * 0.8)
    X_train, y_train = create_sequences(data_scaled[:train_size], sequence_length)
    X_test, y_test = create_sequences(data_scaled[train_size:], sequence_length)
    model_lstm = pipeline.fit(X_train,y_train,lstm__epochs=100, lstm__batch_size=64)
    # model_lstm = build_lstm_model(sequence_length)
    # model_lstm = train_lstm_model(model_lstm, X_train, y_train, epochs=100, batch_size=64)
    r2_lstm, mse_lstm, mae_lstm, rmse_lstm = evaluate_lstm_model(model_lstm, X_test, y_test, scaler)
    save_model_and_scaler(model_lstm, scaler, model_path, scaler_path)
    
    print(f"R² Score (LSTM): {r2_lstm}")
    print(f"MSE (LSTM): {mse_lstm}")
    print(f"MAE (LSTM): {mae_lstm}")
    print(f"RMSE (LSTM): {rmse_lstm}")

