TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "XOM", "KO",
    "^GSPC", "^IXIC"
]

DOWNLOAD = {
    "start": "2016-01-01",
    "period": "10y",
    "interval": "1d",
    "data_dir": "data",
}

FEATURES = {
    "sma_windows": [10, 20, 50],
    "rsi_window": 14,
    "bb_window": 20,
    "macd": (12, 26, 9),
    "volume_ma_window": 20,
}

LABELING = {
    "horizon": 5,
    "threshold": 0.02,
}

SPLIT = {
    "train": 0.70,
    "val":   0.15,
}

SEQUENCE_LEN = 30
