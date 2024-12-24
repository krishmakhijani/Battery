from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import io
import base64
from tqdm import tqdm
import pickle

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

comparison_df = None
comparison_notdf = None

def RandomForestEnteries(num_entries=5000):
    print("\nGenerating random dataset with moderate variations...")
    start_date = datetime.now()
    dates = [start_date + timedelta(minutes=15 * i) for i in range(num_entries)]

    battery_levels = []
    current_level = 80

    for i in range(num_entries):
        change = np.random.normal(0, 1.5)

        if current_level <= 20:
            change = abs(change) * 1.5
        elif current_level >= 90:
            change = -abs(change)

        if np.random.random() < 0.02:
            change *= np.random.choice([-2, 2])

        current_level += change
        current_level = np.clip(current_level, 0, 100)
        battery_levels.append(current_level)

    data = {
        'Timestamp': dates,
        'Battery Level': battery_levels,
        'Network Status': np.random.choice(['wifi', 'cellular'], num_entries, p=[0.7, 0.3]),
        'Power State': ['charging' if level <= 20 else 'unplugged' for level in battery_levels],
        'Bluetooth Connected': np.random.choice(['Yes', 'No'], num_entries, p=[0.6, 0.4]),
        'Location Enabled': np.random.choice(['Yes', 'No'], num_entries, p=[0.8, 0.2]),
        'Brightness': [min(100, max(0, np.random.normal(60, 10))) for _ in range(num_entries)],
        'Total Usage Time': [f"{np.random.randint(0, 30)}m {np.random.randint(0, 59)}s" for _ in range(num_entries)]
    }

    df = pd.DataFrame(data)
    print(f"Generated dataset with {num_entries} entries!")
    return df

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

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size):
        super(xLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention_size = attention_size
        self.attention = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def attention_layer(self, lstm_output):
        attention_scores = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(self.context_vector(attention_scores), dim=1)
        weighted_lstm_output = lstm_output * attention_weights
        context_vector = torch.sum(weighted_lstm_output, dim=1)
        return context_vector

    def forward(self, x):
        lstm_output, (hn, cn) = self.lstm(x)
        attention_output = self.attention_layer(lstm_output)
        attention_output = self.dropout(attention_output)
        out = self.fc(attention_output)
        return out

def create_day_sequences(data, timestamps):
    xs, ys, ts = [], [], []
    for i in range(47, len(data), 48):
        x = data[i-47:i, :]
        y = data[i, 0]
        xs.append(x)
        ys.append(y)
        ts.append(timestamps[i])
    return np.array(xs), np.array(ys), ts

def load_scaler(scaler_file_path):
    try:
        with open(scaler_file_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_file_path}")
        return scaler
    except FileNotFoundError:
        print("No existing scaler found. Creating a new one.")
        return MinMaxScaler()

def save_scaler(scaler, scaler_file_path):
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_file_path}")

def load_model_metadata(metadata_file_path):
    try:
        with open(metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded model metadata from {metadata_file_path}")
        return metadata
    except FileNotFoundError:
        print(f"No existing metadata found at {metadata_file_path}.")
        return None

def save_model(model, file_path, metadata_file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")
    metadata = {
        "input_size": model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "attention_size": model.attention_size,
    }
    with open(metadata_file_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Model metadata saved to {metadata_file_path}")

@app.route('/', methods=['POST'])
def upload_file():
    try:
        data = RandomForestEnteries(5000)
        data.set_index('Timestamp', inplace=True)

        data['Total Usage Time'] = data['Total Usage Time'].apply(convert_duration_to_seconds)

        network_status_mapping = {'wifi': True, 'cellular': False}
        power_state_mapping = {'charging': True, 'unplugged': False}
        bluetooth_connected_mapping = {'Yes': True, 'No': False}
        location_enabled_mapping = {'Yes': True, 'No': False}

        data['Network Status'] = data['Network Status'].map(network_status_mapping)
        data['Power State'] = data['Power State'].map(power_state_mapping)
        data['Bluetooth Connected'] = data['Bluetooth Connected'].map(bluetooth_connected_mapping)
        data['Location Enabled'] = data['Location Enabled'].map(location_enabled_mapping)

        cleaned_data = pd.get_dummies(data, columns=['Network Status', 'Power State', 'Bluetooth Connected', 'Location Enabled'], drop_first=True)

        scaler = load_scaler('scaler.pkl')
        data_scaled = scaler.fit_transform(cleaned_data)
        save_scaler(scaler, 'scaler.pkl')

        X, y, timestamps = create_day_sequences(data_scaled, data.index)
        X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(X, y, timestamps, test_size=0.2, shuffle=False, random_state=42)

        metadata = load_model_metadata('xlstm_metadata.pkl')
        if metadata:
            input_size = metadata['input_size']
            hidden_size = metadata['hidden_size']
            attention_size = metadata['attention_size']
            print("Reusing model architecture from saved metadata.")
        else:
            input_size = len(cleaned_data.columns)
            hidden_size = 64
            attention_size = 32
            print("No metadata found. Using default architecture.")

        model = xLSTM(input_size, hidden_size, attention_size)
        try:
            model.load_state_dict(torch.load('xlstm_weights.pth'))
            print("Loaded existing model weights from xlstm_weights.pth")
        except FileNotFoundError:
            print("No existing model weights found. Training from scratch.")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        if metadata is None:
            print("\nTraining xLSTM model:")
            pbar = tqdm(range(300), desc='Training', ncols=100)
            for epoch in pbar:
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                pbar.set_description(f'Epoch {epoch+1}/300 | Loss: {loss.item():.4f}')
            pbar.close()
            save_model(model, 'xlstm_weights.pth', 'xlstm_metadata.pkl')

        global comparison_df, comparison_notdf
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
            'message': 'Processing completed successfully'
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def process_and_predict():
    if comparison_df is None or comparison_notdf is None:
        return jsonify({'error': 'No data processed yet. Please upload a file first.'}), 400

    print("\nGenerating visualization...")
    plt.figure(figsize=(16, 8))
    plt.plot(comparison_df['Timestamp'], comparison_df['True Battery Level (%)'],
             label='True Battery Level', color='blue', linewidth=1, marker='o', markersize=2)
    plt.plot(comparison_df['Timestamp'], comparison_df['Predicted Battery Level (%)'],
             label='Predicted Battery Level', color='orange', linewidth=1, marker='o', markersize=2)
    plt.title('True vs Predicted Battery Levels with Attention-based xLSTM')
    plt.xlabel('Timestamp')
    plt.ylabel('Battery Level (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=300)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()

    return jsonify({
        'comparison': comparison_notdf.to_dict(orient='records'),
        'image_base64': img_base64
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
