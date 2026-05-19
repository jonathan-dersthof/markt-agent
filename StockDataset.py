import torch
import pandas as pd
from constants import SEQUENCE_LEN


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, sequence_len: int = SEQUENCE_LEN, is_train: bool = False):
        self.data = dataframe.drop(columns=["ticker"], errors="ignore").copy()
        self.features = self.data.drop(columns=["label"]).values
        self.labels = self.data["label"].values
        self.sequence_len = sequence_len
        self.is_train = is_train

    def __len__(self):
        return len(self.data) - self.sequence_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.sequence_len]
        y = self.labels[idx + self.sequence_len]

        x_tensor = torch.tensor(x, dtype=torch.float32)

        if self.is_train:
            noise = torch.randn_like(x_tensor) * 0.05
            x_tensor = x_tensor + noise

        return x_tensor, torch.tensor(y, dtype=torch.long)
