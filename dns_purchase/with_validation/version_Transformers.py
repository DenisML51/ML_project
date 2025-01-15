import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
import random

config = {
    "model": {
        "input_dim": 7,  # Входная размерность
        "num_layers": 3,  # Количество слоев
        "dropout": 0.2,  # Вероятность нулевого значения
        "hidden_dim": 80  # Количество скрытых узлов
    },
    "training": {
        "batch_size": 16,  # Размерность батча
        "epoch": 400,  # Количество эпох обучения
        "learning_rate": 0.00007,  # Скорость обучения
        "sequence_length": 12,  # Длина последовательности
        "early_stopping_patience": 50,  # Ожидание улучшения ошибки
        "loss_target": 0.1,  # Целевая ошибка
        'train_test_split': 0.2
    },
    "forecast": {
        "forecast_weeks": 30,  # Количество недель в прогнозе
        "iterations": 0,  # Прогнозные итерации
        "weeks_behind": 8  # Количество недель назад для прогноза
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
    data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int)
    data['day_of_week'] = data['date'].dt.weekday

    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['week_sin'] = np.sin(2 * np.pi * data['week_of_year'] / 52)
    data['week_cos'] = np.cos(2 * np.pi * data['week_of_year'] / 52)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    scaler = MinMaxScaler()
    data['scaled_product_count'] = scaler.fit_transform(data[['product_count']])


    return data, scaler


def create_date_features(data):
    sequence_length = config['training']['sequence_length']
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        seq_data = data.iloc[i:i + sequence_length][[
            'scaled_product_count',
            'month_sin',
            'month_cos',
            'week_sin',
            'week_cos',
            'day_sin',
            'day_cos']].values

        target = data.iloc[i + sequence_length]['scaled_product_count']
        sequences.append(seq_data)
        targets.append(target)
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def create_dataloader(sequences, targets, batch_size):
    class TimeSeriesDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]

    dataset = TimeSeriesDataset(sequences, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size=1, dropout=0.2, nhead=4):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(hidden_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, output_size)
        self.relu = nn.ReLU()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
        nn.init.uniform_(self.positional_encoding, -0.1, 0.1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

        x = self.transformer(x)

        x = x[:, -1, :]

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.relu(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, writer=None, early_stopping_patience=config['training']['early_stopping_patience']):
    best_loss = float('inf')
    no_improvement_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss_train = 0
        epoch_loss_test = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred.squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()

            if writer:
                writer.add_scalar("Batch Train Loss", loss.item(), epoch * len(train_loader) + batch_idx)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                y_pred = model(inputs)
                loss = criterion(y_pred.squeeze(-1), targets)
                epoch_loss_test += loss.item()

            if writer:
                writer.add_scalar("Batch Validation Loss", loss.item(), epoch * len(test_loader) + batch_idx)


        for name, weight in model.named_parameters():
            writer.add_histogram(f'{name}.weight', weight.detach().numpy(), epoch)
            writer.add_histogram(f'{name}.grad', weight.grad.detach().numpy(), epoch)

        average_loss = epoch_loss_test / len(test_loader)
        average_loss_train = epoch_loss_train / len(train_loader)


        if writer:
            writer.add_scalar("Epoch Validation Loss", average_loss, epoch)
            writer.add_scalar("Epoch Train Loss", average_loss_train, epoch)

        # log_message(f"Эпоха {epoch+1}/{epochs}, Средний Loss: {average_loss:.4f}", "INFO")

        if average_loss < best_loss:
            # log_message(
            #     f"Эпоха {epoch + 1}/{epochs}, AVG_LOSS_test: {average_loss:.4f}, 'AVG_LOSS_train: {average_loss_train:.4f}",
            #     "INFO")

            best_loss = average_loss
            torch.save(model.state_dict(), 'model.pth')
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            # log_message(f"Loss не улучшился {no_improvement_counter} эпох подряд", "WARNING")

        if no_improvement_counter >= early_stopping_patience and best_loss < config['training']['early_stopping_patience']:
            log_message(f"Ранняя остановка на эпохе {epoch+1} из-за отсутствия улучшений", "SUCCESS")
            log_message(
                f"Эпоха {epoch + 1}/{epochs}, AVG_LOSS_test: {best_loss:.4f}, 'AVG_LOSS_train: {average_loss_train:.4f}",
                "INFO")
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

def forecast(df, model, scaler, sequence_length, forecast_weeks=60, freq='MS'):
    """
    Функция для построения прогноза на 8 недель назад и 60 недель вперед, формируя единый прогноз.

    :param freq:
    :param df: DataFrame с исходными данными.
    :param model: Обученная модель LSTM.
    :param scaler: Объект MinMaxScaler для обратного преобразования масштабированных данных.
    :param sequence_length: Длина последовательности для модели.
    :param forecast_weeks: Количество недель для прогноза в будущее.
    :return: DataFrame с датами и прогнозами product_count.
    """
    # Препроцессинг данных
    sequence_length = config['training']['sequence_length']

    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data['product_count'] = data['product_count'].ffill()

    # Создание признаков
    data['month'] = data['date'].dt.month
    data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int)
    data['day_of_week'] = data['date'].dt.weekday
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['week_sin'] = np.sin(2 * np.pi * data['week_of_year'] / 52)
    data['week_cos'] = np.cos(2 * np.pi * data['week_of_year'] / 52)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    # Масштабирование данных
    data['scaled_product_count'] = scaler.transform(data[['product_count']])

    # Начальная точка прогноза (8 недель назад)
    start_idx = len(data['date']) - sequence_length - 8
    recent_data = data.iloc[start_idx:]

    # Прогнозирование
    model.eval()
    predictions = []
    current_sequence = recent_data.iloc[:sequence_length][[
        'scaled_product_count',
        'month_sin', 'month_cos',
        'week_sin', 'week_cos',
        'day_sin', 'day_cos']].values

    # Даты для прогноза (8 недель назад + 60 недель вперед)
    future_dates = pd.date_range(start=recent_data['date'].iloc[sequence_length - 1],
                                 periods=8 + forecast_weeks, freq=freq)

    for i in range(8 + forecast_weeks):
        input_sequence = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_sequence).item()
        predictions.append(pred)

        if i < 8:
            # Для первых 8 недель обновляем последовательность на основе реального временного ряда
            next_row = recent_data.iloc[sequence_length + i]
            next_features = [
                pred,
                next_row['month_sin'], next_row['month_cos'],
                next_row['week_sin'], next_row['week_cos'],
                next_row['day_sin'], next_row['day_cos']
            ]
        else:
            # Для будущих недель обновляем последовательность на основе прогнозируемых данных
            next_date = future_dates[i]
            next_features = [
                pred,
                np.sin(2 * np.pi * next_date.month / 12),
                np.cos(2 * np.pi * next_date.month / 12),
                np.sin(2 * np.pi * next_date.isocalendar().week / 52),
                np.cos(2 * np.pi * next_date.isocalendar().week / 52),
                np.sin(2 * np.pi * next_date.weekday() / 7),
                np.cos(2 * np.pi * next_date.weekday() / 7)
            ]

        current_sequence = np.vstack([current_sequence[1:], next_features])

    # Обратное масштабирование
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Итоговый датафрейм
    result_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted': predictions
    })

    return result_df


def run_pipeline(df,
                 sequence_length=config['training']['sequence_length'],
                 hidden_dim=config['model']['hidden_dim'],
                 num_layers=config['model']['num_layers'],
                 dropout=config['model']['dropout'],
                 learning_rate=config['training']['learning_rate'],
                 epochs=config['training']['epoch'],
                 forecast_weeks=config['forecast']['forecast_weeks'],
                 batch_size=config['training']['batch_size'],
                 split=config['training']['train_test_split'],
                 category=1, freq='MS'):

    writer = SummaryWriter()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    input_dim = config['model']['input_dim']

    delta = 0
    delta += 1
    data, scaler = preprocess_data(df)
    sequences, targets = create_date_features(data)

    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=split, shuffle=False)

    data_loader = create_dataloader(sequences, targets, batch_size)
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size)

    model = Transformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )


    # inputs = torch.randn(batch_size, sequence_length, input_dim)
    # scripted_model = torch.jit.script(model)
    # writer.add_graph(scripted_model, inputs)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    loss = train_model(model, train_loader, test_loader, criterion, optimizer, epochs, writer)

    preds_data, act_data = evaluate_model(model, data_loader, scaler)
    preds_train, act_train = evaluate_model(model, train_loader, scaler)
    preds_test, act_test = evaluate_model(model, test_loader, scaler)

    forecast_df = forecast(df, model, scaler, sequence_length, forecast_weeks, freq=freq)


    plt.rcParams.update({
        'axes.facecolor': '#1E1E1E',  # Фон осей
        'axes.edgecolor': '#ffffff',  # Цвет рамки осей
        'text.color': '#ffffff',  # Цвет текста
        'xtick.color': '#ffffff',  # Цвет подписей оси X
        'ytick.color': '#ffffff',  # Цвет подписей оси Y
        'grid.color': '#1E1E1E',  # Цвет сетки
        'figure.facecolor': '#1E1E1E',  # Фон всей фигуры
        'figure.edgecolor': '#2b2b2b'
    })

    fig, axs = plt.subplots(4, figsize=(15, 15))
    fig.suptitle('Итерация 0')
    fig.patch.set_visible(False)

    axs[0].plot(act_train, label="Actual Train", color="#37B6CE")
    axs[0].plot(preds_train, label="Predicted Train", color="#FF9F40", linestyle="dashed")
    axs[0].legend()

    axs[1].plot(act_test, label="Actual Test", color="orange")
    axs[1].plot(preds_test, label="Predicted Test", color="green", linestyle="dashed")
    axs[1].legend()

    axs[2].plot(act_data, label='Actual Data', color="#37B6CE")
    axs[2].plot(preds_data, label='Predicted Data', color="#FF9F40", linestyle="dashed")
    axs[2].legend()

    axs[3].plot(data['date'][-40:], data['product_count'][-40:], label='Actual Data', color='#37B6CE')
    axs[3].plot(forecast_df['Date'], forecast_df['Predicted'], color='#FF9F40', linestyle='--')
    axs[3].legend()
    fig.show()

    return model, forecast_df

# model, forecast_df = run_pipeline('count_project_data_week_laptop.csv')
