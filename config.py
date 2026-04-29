"""
Configuration for P2-ETF-VARIATIONAL-AUTOENCODER engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-variational-autoencoder-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features (only those fully available from 2008) ---
# Removed IG_SPREAD and HY_SPREAD because they start in 2023
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- VAE Parameters ---
LATENT_DIM = 16
HIDDEN_LAYERS = [512, 256, 128]
BETA = 0.3
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252

# --- Inference ---
NUM_SAMPLES = 500
REGIME_WINDOW = 126

# --- Training Epochs ---
DAILY_EPOCHS = 300
GLOBAL_EPOCHS = 500
SHRINKING_EPOCHS = 250

# --- Training Modes ---
DAILY_LOOKBACK = 1008
GLOBAL_TRAIN_START = "2008-01-01"
# Start shrinking windows from 2008 (or 2009 if you want full 3‑year windows)
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
