import yfinance as yf
import pandas as pd
import numpy as np
import copy

from pathlib import Path
from constants import TICKERS, DOWNLOAD, FEATURES, LABELING

class Importer:
    def __init__(self):
        self.data_dir : Path = Path(DOWNLOAD["data_dir"])
        self.raw_data : dict[str, pd.DataFrame] = self.download_all()

    @staticmethod
    def download_all() -> dict[str, pd.DataFrame]:
        Path(DOWNLOAD["data_dir"]).mkdir(parents=True, exist_ok=True)
        data_dict = {}

        print("Daten herunterladen:")

        for ticker in TICKERS:
            download_path = Path(DOWNLOAD["data_dir"]) / f"{ticker.replace('^','')}.csv"

            if not download_path.exists():
                print(f"    '{ticker}' wird heruntergeladen")

                dataframe = yf.Ticker(ticker).history(
                    start = DOWNLOAD["start"],
                    period = DOWNLOAD["period"],
                    interval = DOWNLOAD["interval"],
                    auto_adjust = True
                )
                dataframe.index = dataframe.index.tz_localize(None)
                dataframe.to_csv(download_path)

                print(f"    {len(dataframe)} Tage gespeichert unter {download_path}")
            else:
                print(f"    '{ticker}' bereits heruntergeladen")

            data_dict[ticker] = pd.read_csv(download_path)

        print(f"\nAlle Daten heruntergeladen unter '{Path(DOWNLOAD["data_dir"])}'")

        return data_dict

    def add_features(self) -> pd.DataFrame:
        all_processed_dfs = []
        config = FEATURES

        for ticker, raw_dataframe in self.raw_data.items():
            dataframe = raw_dataframe.copy()
            close = dataframe["Close"]

            dataframe["return_1d"] = close.pct_change()

            for window in config["sma_windows"]:
                sma = close.rolling(window).mean()
                dataframe[f"sma_{window}"] = sma
                dataframe[f"dist_sma_{window}"] = (close - sma) / sma

            std = close.rolling(config["bb_window"]).std()
            sma20 = close.rolling(config["bb_window"]).mean()
            dataframe["bb_upper"] = sma20 + 2 * std
            dataframe["bb_lower"] = sma20 - 2 * std
            dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / sma20
            dataframe["bb_pos"] = (close - dataframe["bb_lower"]) / (dataframe["bb_upper"] - dataframe["bb_lower"])

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(config["rsi_window"]).mean()
            loss = (-delta).clip(lower=0).rolling(config["rsi_window"]).mean()
            dataframe["rsi"] = 100 - (100 / (1 + gain / loss))

            f, s, sig = config["macd"]
            ema_fast = close.ewm(span=f, adjust=False).mean()
            ema_slow = close.ewm(span=s, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            dataframe["macd"] = macd_line
            dataframe["macd_signal"] = macd_line.ewm(span=sig, adjust=False).mean()
            dataframe["macd_hist"] = macd_line - dataframe["macd_signal"]

            vol_ma = dataframe["Volume"].rolling(config["volume_ma_window"]).mean()
            dataframe["volume_ratio"] = dataframe["Volume"] / vol_ma

            dataframe["ticker"] = ticker

            dataframe.drop(columns=["Open", "High", "Low", "Volume",
                             "Dividends", "Stock Splits",
                             *[f"sma_{w}" for w in config["sma_windows"]]],
                    errors="ignore", inplace=True)

            dataframe.dropna(inplace=True)
            all_processed_dfs.append(dataframe)

        final_df = pd.concat(all_processed_dfs)

        return final_df

if __name__ == "__main__":
    importer = Importer()
    print(importer.raw_data)
