import torch
import torch.nn.functional as f
import pandas as pd

from sklearn.preprocessing import StandardScaler

from constants import SEQUENCE_LEN


class Predictor:
    def __init__(self, model, scaler: StandardScaler, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model.eval()

        self.scaler = scaler

    def predict(self, dataframe : pd.DataFrame, index : int):
        if index < SEQUENCE_LEN:
            index = SEQUENCE_LEN

        if len(dataframe) < SEQUENCE_LEN:
            raise ValueError(f"Nicht genug Daten. Mindestens {SEQUENCE_LEN} Tage benötigt.")

        data = dataframe.iloc[index - SEQUENCE_LEN : index].copy()

        feature_cols = [col for col in data.columns if col not in ["ticker", "label"]]
        features_raw = data[feature_cols].values

        scaled_features = self.scaler.transform(features_raw)

        tensor_x = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probabilities = f.softmax(logits, dim=1).squeeze().cpu().numpy()

        return {
            "Down (0)": round(probabilities[0] * 100, 2),
            "Neutral (1)": round(probabilities[1] * 100, 2),
            "Up (2)": round(probabilities[2] * 100, 2)
        }

    def predict_latest(self, dataframe: pd.DataFrame):
        return self.predict(dataframe, len(dataframe))
