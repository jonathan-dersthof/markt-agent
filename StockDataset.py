import torch
import pandas as pd
from constants import SEQUENCE_LEN


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, sequence_len: int = SEQUENCE_LEN):
        self.data = dataframe.drop(columns=["ticker"], errors="ignore").copy()
        self.features = self.data.drop(columns=["label"]).values
        self.labels = self.data["label"].values
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.data) - self.sequence_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.sequence_len]
        y = self.labels[idx + self.sequence_len]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
