# !pip install flask-cors flask pandas numpy torch matplotlib scikit-learn

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import threading
import os


matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

comparison_df = None
comparison_notdf = None

def process_uploaded_data(df):
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    network_status_mapping = {'wifi': True, 'cellular': False}
    power_state_mapping = {'charging': True, 'unplugged': False}
    bluetooth_connected_mapping = {'Yes': True, 'No': False}
    location_enabled_mapping = {'Yes': True, 'No': False}

    df['Network Status'] = df['Network Status'].map(network_status_mapping)
    df['Power State'] = df['Power State'].map(power_state_mapping)
    df['Bluetooth Connected'] = df['Bluetooth Connected'].map(bluetooth_connected_mapping)
    df['Location Enabled'] = df['Location Enabled'].map(location_enabled_mapping)

    X = df[['Hour', 'DayOfWeek', 'Battery Level', 'Network Status', 'Power State', 'Bluetooth Connected', 'Location Enabled', 'Brightness']]
    y = df['Total Usage Time']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return df, rf_model

def generate_next_entry(last_entry, rf_model):
    next_timestamp = last_entry['Timestamp'] + timedelta(minutes=30)
    battery_level = min(100, last_entry['Battery Level'] + np.random.randint(1, 5)) if last_entry['Power State'] else max(0, last_entry['Battery Level'] - np.random.randint(1, 5))

    network_status = np.random.choice([True, False], p=[0.7, 0.3])
    power_state = True if battery_level < 20 else False
    bluetooth_connected = np.random.choice([True, False], p=[0.4, 0.6])
    location_enabled = np.random.choice([True, False], p=[0.9, 0.1])
    brightness = round(max(0, min(100, np.random.normal(last_entry['Brightness'], 5))), 2)

    rf_input = np.array([[next_timestamp.hour, next_timestamp.dayofweek, battery_level, network_status, power_state, bluetooth_connected, location_enabled, brightness]])
    total_usage_time = rf_model.predict(rf_input)[0]

    return pd.Series({
        'Timestamp': next_timestamp,
        'Battery Level': battery_level,
        'Network Status': network_status,
        'Power State': power_state,
        'Bluetooth Connected': bluetooth_connected,
        'Location Enabled': location_enabled,
        'Brightness': brightness,
        'Total Usage Time': total_usage_time
    })

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size):
        super(xLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention_size = attention_size
        self.attention = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size, 1)

    def attention_layer(self, lstm_output):
        attention_scores = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(self.context_vector(attention_scores), dim=1)
        weighted_lstm_output = lstm_output * attention_weights
        context_vector = torch.sum(weighted_lstm_output, dim=1)
        return context_vector

    def forward(self, x):
        lstm_output, (hn, cn) = self.lstm(x)
        attention_output = self.attention_layer(lstm_output)
        out = self.fc(attention_output)
        return out

def create_day_sequences(data, timestamps):
    xs, ys, ts = [], [], []
    for i in range(23, len(data), 24):
        x = data[i-23:i, :]
        y = data[i, 0]
        xs.append(x)
        ys.append(y)
        ts.append(timestamps[i])
    return np.array(xs), np.array(ys), ts

def convert_duration_to_seconds(duration):
    if isinstance(duration, str):
        minutes = seconds = 0
        parts = duration.split(' ')
        for part in parts:
            if 'm' in part:
                minutes = int(part.replace('m', ''))
            elif 's' in part:
                seconds = int(part.replace('s', ''))
        return minutes * 60 + seconds
    return 0

@app.route('/', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        file_path = './data.csv'
        file.save(file_path)

        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        processed_df, rf_model = process_uploaded_data(df)

        num_new_entries = 500
        new_data = []
        last_entry = processed_df.iloc[-1]

        for i in range(num_new_entries):
            new_entry = generate_next_entry(last_entry, rf_model)
            new_data.append(new_entry)
            last_entry = new_entry

        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([processed_df, new_df], ignore_index=True)
        combined_df = combined_df.drop(columns=['DayOfWeek', 'Hour'])

        combined_df.to_csv('./generated_data.csv', index=False)

        global comparison_df, comparison_notdf

        data = pd.read_csv('./generated_data.csv')
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)

        data['Total Usage Time'] = data['Total Usage Time'].apply(convert_duration_to_seconds)
        cleaned_data = pd.get_dummies(data, columns=['Network Status', 'Power State', 'Bluetooth Connected', 'Location Enabled'], drop_first=True)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(cleaned_data)
        data_scaled = pd.DataFrame(data_scaled, columns=cleaned_data.columns, index=cleaned_data.index)

        X, y, timestamps = create_day_sequences(data_scaled.values, data.index)
        X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(X, y, timestamps, test_size=0.2, shuffle=False, random_state=42)

        input_size = len(cleaned_data.columns)
        hidden_size = 64
        attention_size = 32
        model = xLSTM(input_size, hidden_size, attention_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        num_epochs = 250
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)

        y_pred_inverse = scaler.inverse_transform(
            np.concatenate((y_pred.numpy(), np.zeros((y_pred.shape[0], len(cleaned_data.columns) - 1))), axis=1)
        )[:, 0]
        y_test_inverse = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(cleaned_data.columns) - 1))), axis=1)
        )[:, 0]

        y_pred_inverse = np.clip(y_pred_inverse, 0, 100)
        y_test_inverse = np.clip(y_test_inverse, 0, 100)

        comparison_notdf = pd.DataFrame({
            'True Battery Level (%)': y_test_inverse,
            'Predicted Battery Level (%)': y_pred_inverse
        })

        comparison_df = pd.DataFrame({
            'Timestamp': ts_test,
            'True Battery Level (%)': y_test_inverse,
            'Predicted Battery Level (%)': y_pred_inverse
        })

        return jsonify({
            'message': 'File processed successfully',
            'generated_data_path': './generated_data.csv'
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def process_and_predict():
    if comparison_df is None or comparison_notdf is None:
        return jsonify({'error': 'No data processed yet. Please upload a file first.'}), 400

    plt.figure(figsize=(14, 7))
    plt.plot(comparison_df['Timestamp'], comparison_df['True Battery Level (%)'],
             label='True Battery Level', color='blue')
    plt.plot(comparison_df['Timestamp'], comparison_df['Predicted Battery Level (%)'],
             label='Predicted Battery Level', color='orange')
    plt.title('True vs Predicted Battery Levels with Attention-based xLSTM')
    plt.xlabel('Timestamp')
    plt.ylabel('Battery Level (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a buffer instead of displaying it
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()

    return jsonify({
        'comparison': comparison_notdf.to_dict(orient='records'),
        'image_base64': img_base64
    })

def run_app():
    print("\n\u1F680 Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)

run_app()
