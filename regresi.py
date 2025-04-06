import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io
import base64

def load_and_train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url)
    df = df[['Temp']]
    df['Humidity'] = 100 - df['Temp'] * 2  # Simulasi data

    X = df[['Humidity']].values
    y = df[['Temp']].values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_scaled, y_scaled, epochs=100, verbose=0)

    return model, scaler_x, scaler_y

def predict_temperature(humidity, model, scaler_x, scaler_y):
    humidity_scaled = scaler_x.transform([[humidity]])
    pred_scaled = model.predict(humidity_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(pred_scaled)
    return round(prediction[0][0], 2)

def generate_plot(model, scaler_x, scaler_y):
    x_values = np.linspace(30, 90, 100).reshape(-1, 1)
    x_scaled = scaler_x.transform(x_values)
    y_pred_scaled = model.predict(x_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_pred, color='red', label='Prediksi')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Temperature (°C)')
    plt.title('Regresi Linear: Humidity vs Temp')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{encoded}"


