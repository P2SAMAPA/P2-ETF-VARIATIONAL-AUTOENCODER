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

# --- VAE Parameters (higher complexity) ---
LATENT_DIM = 12                      # more expressive latent space
HIDDEN_LAYERS = [256, 128, 64]       # deeper encoder/decoder
BETA = 0.5                           # β-VAE weight for KL loss
LEARNING_RATE = 0.001
BATCH_SIZE = 128
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252

# --- Inference ---
NUM_SAMPLES = 200                    # more Monte Carlo samples
REGIME_WINDOW = 63

# --- Training Epochs ---
DAILY_EPOCHS = 100                   # was 50
GLOBAL_EPOCHS = 150                  # was 100
SHRINKING_EPOCHS = 80                # was 40

# --- Training Modes ---
DAILY_LOOKBACK = 1008                # 4 years (was 504 days → too few samples)
GLOBAL_TRAIN_START = "2008-01-01"
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
