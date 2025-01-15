import pandas as pd
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset

def preprocess_data(df):
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data['product_count'] = data['product_count'].ffill()

    data_shifted = data['product_count'].shift(periods=1)
    data_diff = data['product_count'] - data_shifted

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


def create_date_features(data, sequence_length, split_ration=0.8):
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

    split_index = int(len(sequences) * split_ration)
    train_sequences = np.array(sequences[:split_index]).astype(np.float32)
    train_targets = np.array(targets[:split_index]).astype(np.float32)
    val_sequences = np.array(sequences[split_index:]).astype(np.float32)
    val_targets = np.array(targets[split_index:]).astype(np.float32)
    return train_sequences, train_targets, val_sequences, val_targets


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

df = pd.read_csv('data_month_dep_2.csv')
df = df.where(df.category_id == 'fd721048-70e6-11e2-b24e-00155d030b1f').dropna()
df.drop(['category_id', 'product_cost'], axis=1, inplace=True)

print(df.info())
data, scaler = preprocess_data(df)
train_sequences, train_targets, val_sequences, val_targets = create_date_features(data, 12)
print(train_sequences)
print(train_targets)
