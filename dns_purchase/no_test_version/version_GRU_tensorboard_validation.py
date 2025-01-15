import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import optuna

config = {
    "model": {
        "input_dim": 3,  # Входная размерность
        "num_layers": 1,  # Количество слоев
        "dropout": 0,  # Вероятность нулевого значения
        "hidden_dim": 90  # Количество скрытых узлов
    },
    "training": {
        "batch_size": 64,  # Размерность батча
        "epoch": 600,  # Количество эпох обучения
        "learning_rate": 1e-3,  # Скорость обучения
        "sequence_length": 12,  # Длина последовательности
        "early_stopping_patience": 50,  # Ожидание улучшения ошибки
        "loss_target": 0.009,  # Целевая ошибка
        "optuna_chech": 0
    },
    "forecast": {
        "forecast_weeks": 26,  # Количество недель в прогнозе
        "iterations": 0,  # Прогнозные итерации
        "weeks_behind": -2  # Количество недель назад для прогноза
    }
}

def log_message(message, level="INFO"):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] [{level}] {message}")

def preprocess_data(df):
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data['product_count'] = data['product_count'].ffill()

    data['month'] = data['date'].dt.month

    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    scaler = MinMaxScaler()
    data['scaled_product_count'] = scaler.fit_transform(data[['product_count']])

    return data, scaler

def create_date_features(data, sequence_length, split_ration=0.8):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        seq_data = data.iloc[i:i + sequence_length][[
            'scaled_product_count',
            'month_sin',
            'month_cos']].values

        target = data.iloc[i + sequence_length]['scaled_product_count']
        sequences.append(seq_data)
        targets.append(target)

    split_index = int(len(sequences) * split_ration)
    train_sequences = np.array(sequences[:split_index]).astype(np.float32)
    train_targets = np.array(targets[:split_index]).astype(np.float32)
    val_sequences = np.array(sequences[split_index:]).astype(np.float32)
    val_targets = np.array(targets[split_index:]).astype(np.float32)
    return train_sequences, train_targets, val_sequences, val_targets

def create_dataset(sequences, targets):
    class TimeSeriesDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]

    dataset = TimeSeriesDataset(sequences, targets)
    return dataset

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, writer=None, early_stopping_patience=config['training']['early_stopping_patience']):
    best_loss = float('inf')
    no_improvement_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1), targets)
                epoch_val_loss += loss.item()


        for name, weight in model.named_parameters():
            writer.add_histogram(f'{name}.weight', weight.detach().numpy(), epoch)
            writer.add_histogram(f'{name}.grad', weight.grad.detach().numpy(), epoch)

        average_loss = epoch_val_loss / len(val_loader)

        if writer:
            writer.add_scalar("Epoch Loss", average_loss, epoch)

        log_message(f"Эпоха {epoch+1}/{epochs}, Средний Loss: {average_loss:.4f}", "INFO")

        if average_loss < best_loss:
            best_loss = average_loss
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            log_message(f"Loss не улучшился {no_improvement_counter} эпох подряд", "WARNING")

        if no_improvement_counter >= early_stopping_patience and best_loss < config['training']['early_stopping_patience']:
            log_message(f"Ранняя остановка на эпохе {epoch+1} из-за отсутствия улучшений", "SUCCESS")
            break

    return loss

def evaluate_model(model, train_loader, scaler):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():

        for inputs, targets in train_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze(-1).tolist())
            actuals.extend(targets.tolist())
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    return predictions, actuals

def make_forecast(model, last_sequence, scaler, forecast_weeks, last_date):
    forecast_predictions = []
    current_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

    forecast_dates = pd.date_range(start=last_date, periods=forecast_weeks, freq='MS')

    for date in forecast_dates:
        month_sin = np.sin(2 * np.pi * date.month / 12)
        month_cos = np.cos(2 * np.pi * date.month / 12)

        seasonal_features = torch.tensor([month_sin, month_cos], dtype=torch.float32)
        seasonal_features = seasonal_features.unsqueeze(0).unsqueeze(0)
        seasonal_features = seasonal_features.repeat(1, current_sequence.shape[1], 1)

        with torch.no_grad():
            input_with_features = torch.cat((current_sequence[:, :, :-2], seasonal_features), dim=-1)
            output = model(input_with_features)
            forecast_predictions.append(output.item())

            output_expanded = torch.zeros(1, 1, current_sequence.shape[-1])
            output_expanded[:, :, -1] = output
            next_step = torch.cat((current_sequence[:, 1:, :], output_expanded), dim=1)
            current_sequence = next_step

    forecast_predictions = scaler.inverse_transform(np.array(forecast_predictions).reshape(-1, 1)).flatten()
    return pd.DataFrame({
        "Date": forecast_dates,
        "Predicted": forecast_predictions
    })

def run_pipeline(df,
                 sequence_length=config['training']['sequence_length'],
                 hidden_dim=config['model']['hidden_dim'],
                 num_layers=config['model']['num_layers'],
                 dropout=config['model']['dropout'],
                 learning_rate=config['training']['learning_rate'],
                 epochs=config['training']['epoch'],
                 forecast_weeks=config['forecast']['forecast_weeks'],
                 batch_size=config['training']['batch_size'],
                 iterations=config['forecast']['iterations'],
                 category=1):

    writer = SummaryWriter()

    input_dim = config['model']['input_dim']

    h = 0
    delta = 0
    while h != 99:
        delta += 1
        data, scaler = preprocess_data(df)
        train_sequences, train_targets, val_sequences, val_targets = create_date_features(data, sequence_length)
        train_set = create_dataset(train_sequences, train_targets)
        val_set = create_dataset(val_sequences, val_targets)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = GRUNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

        for inputs, targets in train_loader:
            writer.add_graph(model, inputs)
            break

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, writer)
        predictions, actuals = evaluate_model(model, train_loader, scaler)
        pred_val, actual_val = evaluate_model(model, val_loader, scaler)

        forecast_df = make_forecast(model, val_sequences[-1], scaler, forecast_weeks, last_date=data['date'].iloc[config['forecast']['weeks_behind']])

        if len(forecast_df['Predicted'].unique()) <= 5 or loss > config['training']['loss_target']:
            print("Модель в состоянии недообучения")
            if delta >= 30:
                print(f'Невозможно построить прогноз для категории {category}')
                forecast_df = 0
                break
            else:
                continue
        else:
            h = 99
            print('Модель обучена')
            print('Средний LOSS =', loss)

        fig, axs = plt.subplots(4, figsize=(12, 12))
        fig.suptitle('Итерация 0')

        axs[0].plot(actuals, label="Тренировочная выборка", color="blue")
        axs[0].plot(predictions, label="Тренировочный прогноз", color="orange", linestyle="dashed")
        axs[0].legend()
        axs[0].set_title('Обучение')

        axs[3].plot(actual_val, label="Валидация факт", color="purple")
        axs[3].plot(pred_val, label='Валидация прогноз', color='red', linestyle="dashed")

        axs[1].plot(data['date'][-20:], data['product_count'][-20:], label='Actual Data', color='blue')
        axs[1].plot(forecast_df['Date'], forecast_df['Predicted'], label='Forecast', color='orange',
                 linestyle='dashed')
        axs[1].legend()
        axs[1].set_title('Прогноз')

        axs[2].plot(data['date'], data['product_count'], label='Actual Data', color='blue')
        axs[2].plot(forecast_df['Date'], forecast_df['Predicted'], label='Forecast', color='orange',
                 linestyle='dashed')
        axs[2].set_title('Валидация')

        fig.show()

    if iterations > 0:
        forecast_df_sum = pd.DataFrame(columns=['Date', 'Predicted'])

        for i in range(iterations):
            forecast_df_1 = make_forecast(model, val_sequences[-1], scaler, forecast_weeks, last_date=data['date'].iloc[config['forecast']['weeks_behind']])
            forecast_df_sum = pd.merge(forecast_df_sum, forecast_df_1, on='Date')
    else:
        forecast_df = make_forecast(model, val_sequences[-1], scaler, forecast_weeks, last_date=data['date'].iloc[config['forecast']['weeks_behind']])

    print(f'прогноз построен для категории {category}')
    return model, forecast_df

# model, forecast_df = run_pipeline('count_project_data_week_laptop.csv')
