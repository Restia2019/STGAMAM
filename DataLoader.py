import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
from joblib import dump


class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        return 1

    def __getitem__(self, index):
        index += 1
        idx = 1
        start = np.random.randint(0, len(self.df[self.df["reindexed_id"] == idx]) - 50 - self.T - self.S)

        station_number = [str(self.df[self.df["reindexed_id"] == idx][["station_id"]][start:start + 1].values.item())]

        index_in = torch.tensor([i for i in range(start, start + self.T)])

        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])

        _input = torch.tensor(self.df[self.df["reindexed_id"] == idx][
                                  ["temperature", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month",
                                   "cos_month"]][
                              start: start + self.T].values)

        target = torch.tensor(self.df[self.df["reindexed_id"] == idx][
                                  ["temperature", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][
                              start + self.T: start + self.T + self.S].values)

        scaler = self.transform
        scaler.fit(_input[:, 0].unsqueeze(-1))
        _input[:, 0] = torch.tensor(scaler.transform(_input[:, 0].unsqueeze(-1)).squeeze(-1))
        target[:, 0] = torch.tensor(scaler.transform(target[:, 0].unsqueeze(-1)).squeeze(-1))

        dump(scaler, 'scalar_item.joblib')

        for idx in range(2, len(self.df.groupby(by=["reindexed_id"])) + 1):
            station_number.append(
                str(self.df[self.df["reindexed_id"] == idx][["station_id"]][start:start + 1].values.item()))

            t1 = torch.tensor(self.df[self.df["reindexed_id"] == idx][
                                  ["temperature", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month",
                                   "cos_month"]][
                              start: start + self.T].values)

            t2 = torch.tensor(self.df[self.df["reindexed_id"] == idx][
                                  ["temperature", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month",
                                   "cos_month"]][
                              start + self.T: start + self.T + self.S].values)

            scaler.fit(t1[:, 0].unsqueeze(-1))
            t1[:, 0] = torch.tensor(scaler.transform(t1[:, 0].unsqueeze(-1)).squeeze(-1))
            _input = torch.cat([_input, t1], dim=0)
            t2[:, 0] = torch.tensor(scaler.transform(t2[:, 0].unsqueeze(-1)).squeeze(-1))
            target = torch.cat([target, t2], dim=0)

        src_rearrange = []
        for i in range(self.T):
            for j in range(0, len(_input), self.T):
                src_rearrange.append(_input[i + j])
        src_rearrange = torch.cat([t.unsqueeze(0) for t in src_rearrange], dim=0)
        _input = src_rearrange

        target_rearrange = []
        for i in range(self.S):
            for j in range(0, len(target), self.S):
                target_rearrange.append(target[i + j])
        target_rearrange = torch.cat([t.unsqueeze(0) for t in target_rearrange], dim=0)
        target = target_rearrange

        return index_in, index_tar, _input, target, station_number
