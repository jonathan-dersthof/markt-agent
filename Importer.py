import yfinance as yf
import pandas as pd
import numpy as np

from pathlib import Path
from constants import TICKERS, DOWNLOAD, FEATURES, LABELING


class Importer:
    def __init__(self,  data_dir = DOWNLOAD["data_dir"]):
        self.data_dir : Path = data_dir

        self.raw_data : dict[str, pd.DataFrame] | None = None
        self.processed_data : dict[str, pd.DataFrame] | None = None

    def download(self, ticker : str) -> pd.DataFrame:
        (Path(self.data_dir) / "raw").mkdir(parents=True, exist_ok=True)
        download_path = Path(self.data_dir) / "raw" / f"{ticker.replace('^', '')}.csv"

        if not download_path.exists():
            print(f"'{ticker}' wird heruntergeladen")

            dataframe = yf.Ticker(ticker).history(
                start=DOWNLOAD["start"],
                period=DOWNLOAD["period"],
                interval=DOWNLOAD["interval"],
                auto_adjust=True
            )
            dataframe.index = dataframe.index.tz_localize(None)
            dataframe.to_csv(download_path)

            print(f"{len(dataframe)} Tage gespeichert unter {download_path}")
        else:
            dataframe = pd.read_csv(download_path, index_col=0, parse_dates=True)

            print(f"'{ticker}' bereits heruntergeladen")

        return dataframe

    @staticmethod
    def add_features(ticker : str, raw_dataframe : pd.DataFrame) -> pd.DataFrame:
        ticker = ticker.replace('^', '')

        dataframe = raw_dataframe.copy()
        close = dataframe["Close"]

        dataframe["return_1d"] = close.pct_change()

        for window in FEATURES["sma_windows"]:
            sma = close.rolling(window).mean()
            dataframe[f"sma_{window}"] = sma
            dataframe[f"dist_sma_{window}"] = (close - sma) / sma

        std = close.rolling(FEATURES["bb_window"]).std()
        sma20 = close.rolling(FEATURES["bb_window"]).mean()
        dataframe["bb_upper"] = sma20 + 2 * std
        dataframe["bb_lower"] = sma20 - 2 * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / sma20
        dataframe["bb_pos"] = (close - dataframe["bb_lower"]) / (dataframe["bb_upper"] - dataframe["bb_lower"])

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(FEATURES["rsi_window"]).mean()
        loss = (-delta).clip(lower=0).rolling(FEATURES["rsi_window"]).mean()
        dataframe["rsi"] = 100 - (100 / (1 + gain / loss))

        f, s, sig = FEATURES["macd"]
        ema_fast = close.ewm(span=f, adjust=False).mean()
        ema_slow = close.ewm(span=s, adjust=False).mean()

        # MACD als prozentuale Abweichung ausdrücken statt in absoluten Dollar
        macd_line = ((ema_fast - ema_slow) / ema_slow) * 100
        dataframe["macd"] = macd_line
        dataframe["macd_signal"] = macd_line.ewm(span=sig, adjust=False).mean()
        dataframe["macd_hist"] = macd_line - dataframe["macd_signal"]

        vol_ma = dataframe["Volume"].rolling(FEATURES["volume_ma_window"]).mean()
        dataframe["volume_ratio"] = dataframe["Volume"] / vol_ma

        dataframe["ticker"] = ticker

        dataframe.drop(columns=["Open", "High", "Low", "Volume",
                                "Dividends", "Stock Splits",
                                "bb_upper", "bb_lower",
                                *[f"sma_{w}" for w in FEATURES["sma_windows"]]],
                       errors="ignore", inplace=True)

        dataframe.dropna(inplace=True)

        return dataframe

    @staticmethod
    def add_labels(dataframe : pd.DataFrame) -> pd.DataFrame:
        horizon = LABELING["horizon"]
        threshold = LABELING["threshold"]

        future_price = dataframe["Close"].shift(-horizon)

        future_return = (future_price / dataframe["Close"]) - 1

        dataframe["label"] = np.where(
            future_return > threshold, 2,
            np.where(future_return < -threshold, 0, 1)
        )
        dataframe = dataframe[future_return.notna()]

        return dataframe

    def process(self, ticker):
        (Path(self.data_dir) / "processed").mkdir(parents = True, exist_ok = True)
        processed_path = Path(self.data_dir) / "processed" / f"{ticker.replace('^', '')}.csv"

        if not processed_path.exists():
            raw_data = self.download(ticker)
            features = self.add_features(ticker, raw_data)
            processed = self.add_labels(features)
            processed.drop(columns = ["Close"], errors = "ignore", inplace = True)
        else:
            processed = pd.read_csv(processed_path, index_col = 0, parse_dates=True)

        return processed

    def process_all(self) -> dict[str, pd.DataFrame]:
        raw_data_dict = {}
        processed_data_dict = {}

        for ticker in TICKERS:
            raw_path = Path(self.data_dir) / "raw" / f"{ticker.replace('^', '')}.csv"
            processed_path = Path(self.data_dir) / "processed" / f"{ticker.replace('^', '')}.csv"

            processed = self.process(ticker)
            processed.to_csv(processed_path, index=True)
            processed_data_dict[ticker] = processed

            raw_data_dict[ticker] = pd.read_csv(raw_path, index_col=0, parse_dates=True)

        self.processed_data = processed_data_dict
        self.raw_data = raw_data_dict

        return processed_data_dict

if __name__ == "__main__":
    importer = Importer()
    importer.process_all()
    print(importer.processed_data)
