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

# --- Macro Features (conditioning) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- VAE Parameters ---
LATENT_DIM = 8                       # latent space dimension
HIDDEN_LAYERS = [128, 64]            # encoder/decoder hidden layers
BETA = 0.5                           # β-VAE weight for KL loss
LEARNING_RATE = 0.001
EPOCHS = 100                         # training epochs
BATCH_SIZE = 128
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252

# --- Inference ---
NUM_SAMPLES = 100                    # Monte Carlo samples for expected return
REGIME_WINDOW = 63                   # lookback for regime stress computation

# --- Training Modes ---
DAILY_LOOKBACK = 504
GLOBAL_TRAIN_START = "2008-01-01"
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
