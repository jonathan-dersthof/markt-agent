import pandas as pd
import copy
import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset

from constants import SEQUENCE_LEN, SPLIT

from Importer import Importer
from StockDataset import StockDataset
from StockLSTM import StockLSTM
from Predictor import Predictor

class LSTMTrainer:
    def __init__(self,
                 processed_data_dict: dict,
                 sequence_len: int = SEQUENCE_LEN,
                 hidden_size=64,
                 num_layers=1,
                 learning_rate = 0.001,
                 batch_size=32,
                 dropout = 0.3
                 ):
        self.processed_data_dict = processed_data_dict
        self.sequence_len = sequence_len
        self.global_scaler = StandardScaler()
        self.global_datasets = self.setup_datasets()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Training läuft auf: {self.device}")

        sample_df = next(iter(self.processed_data_dict.values()))
        input_size = len(sample_df.columns) - 2

        self.model = StockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=3, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss(weight=self.calculate_class_weights())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)

        self.train_loader = DataLoader(self.global_datasets["train"], batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.global_datasets["val"], batch_size=batch_size, shuffle=False)

    @staticmethod
    def split_and_scale_data(df: pd.DataFrame):
        """ Teilt die Daten auf und skaliert die Features unabhängig pro Ticker """
        n = len(df)
        train_end = int(n * SPLIT["train"])
        val_end = int(n * (SPLIT["train"] + SPLIT["val"]))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()

        feature_cols = [col for col in df.columns if col not in ["ticker", "label"]]

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

        if len(val_df) > 0:
            val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        return train_df, val_df

    def setup_datasets(self):
        """Erstellt den globalen Buffer und fittet EINEN globalen Scaler."""
        train_dfs = []
        val_dfs = []

        for ticker, df in self.processed_data_dict.items():
            n = len(df)
            train_end = int(n * SPLIT["train"])
            val_end = int(n * (SPLIT["train"] + SPLIT["val"]))

            train_dfs.append(df.iloc[:train_end].copy())
            val_dfs.append(df.iloc[train_end:val_end].copy())

        feature_cols = [col for col in train_dfs[0].columns if col not in ["ticker", "label"]]

        combined_train_features = np.vstack([df[feature_cols].values for df in train_dfs])
        self.global_scaler.fit(combined_train_features)

        train_datasets = []
        val_datasets = []

        for t_df in train_dfs:
            t_df[feature_cols] = self.global_scaler.transform(t_df[feature_cols].values)
            if len(t_df) > self.sequence_len:
                train_datasets.append(StockDataset(t_df, self.sequence_len))

        for v_df in val_dfs:
            v_df[feature_cols] = self.global_scaler.transform(v_df[feature_cols].values)
            if len(v_df) > self.sequence_len:
                val_datasets.append(StockDataset(v_df, self.sequence_len))

        return {
            "train": ConcatDataset(train_datasets),
            "val": ConcatDataset(val_datasets)
        }

    def setup_training(self,
                       hidden_size,
                       batch_size,
                       learning_rate,
                       weight_decay,
                       num_layers
                       ):
        sample_df = next(iter(self.processed_data_dict.values()))
        input_size = len(sample_df.columns) - 2

        self.model = StockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=3, )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.train_loader = DataLoader(self.global_datasets["train"], batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.global_datasets["val"], batch_size=batch_size, shuffle=False)

    def calculate_class_weights(self):
        all_labels = []

        for df in self.processed_data_dict.values():
            train_end = int(len(df) * SPLIT["train"])
            all_labels.extend(df["label"].iloc[:train_end].values)

        all_labels = np.array(all_labels)
        class_counts = np.bincount(all_labels, minlength=3)
        total_samples = len(all_labels)

        weights = total_samples / (3.0 * class_counts)

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train_model(self):
        self.model.train()
        total_train_loss = 0

        for states, targets in self.train_loader:
            states, targets = states.to(self.device), targets.to(self.device)
            predictions = self.model(states)
            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()

        return total_train_loss

    def evaluate_model(self):
        self.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for states, targets in self.val_loader:
                states, targets = states.to(self.device), targets.to(self.device)
                predictions = self.model(states)

                loss = self.criterion(predictions, targets)
                total_val_loss += loss.item()

                predicted_classes = torch.max(predictions, dim=1)[1]
                correct_predictions += (predicted_classes == targets).sum().item()
                total_samples += targets.size(0)

        val_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

        return total_val_loss, val_accuracy

    def run_epoch(self):
        total_train_loss = self.train_model()
        total_val_loss, val_accuracy = self.evaluate_model()

        return total_train_loss, total_val_loss, val_accuracy

    def train(self, epochs=20):
        best_val_loss = np.inf
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch = 0

        for epoch in range(epochs):
            total_train_loss, total_val_loss, val_accuracy = self.run_epoch()

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(self.model.state_dict())

            print(f"Epoch [{epoch + 1}/{epochs}] | "
                  f"Train Loss: {total_train_loss:.4f} | "
                  f"Val Loss: {total_val_loss:.4f} | "
                  f"Val Accuracy: {val_accuracy:.2f}%")

        print(f"\nTraining beendet! Bestes Modell in Epoche {best_epoch} mit Val Loss: {best_val_loss:.4f} geladen.")
        self.model.load_state_dict(best_model_wts)

        return self.model

if __name__ == "__main__":
    importer = Importer()
    processed_dict = importer.process_all()

    for ticker, df in processed_dict.items():
        print(f"Verteilung für {ticker}:")
        print(df["label"].value_counts(normalize=True))  # Zeigt Prozentanteile

    trainer = LSTMTrainer(processed_dict)
    trained_model_a = trainer.train(epochs=30)
    trainer.setup_training(32, 32, 0.003, 1e-3, 1)
    trained_model_b = trainer.train(epochs=30)

    predicted = []

    predictor_a = Predictor(trained_model_a, trainer.global_scaler)
    predictor_b = Predictor(trained_model_b, trainer.global_scaler)

    predicted.append(predictor_a.predict(processed_dict["AAPL"], 1000))
    predicted.append(predictor_b.predict(processed_dict["AAPL"], 1000))
    print(processed_dict["AAPL"].iat[1000, 12])

    predicted.append(predictor_a.predict(processed_dict["AAPL"], 1050))
    predicted.append(predictor_b.predict(processed_dict["AAPL"], 1050))
    print(processed_dict["AAPL"].iat[1050, 12])

    for p in predicted:
        print(p)
