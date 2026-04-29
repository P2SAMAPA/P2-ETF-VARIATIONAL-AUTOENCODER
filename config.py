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

# --- Expanded Macro Features (more conditioning) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# --- VAE Parameters (maximum complexity) ---
LATENT_DIM = 16                      # larger latent space
HIDDEN_LAYERS = [512, 256, 128]      # very deep encoder/decoder
BETA = 0.3                           # lower KL weight → better reconstruction
LEARNING_RATE = 0.0005               # lower LR for stable deep training
BATCH_SIZE = 256                     # larger batch for GPU efficiency
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252

# --- Inference ---
NUM_SAMPLES = 500                    # more MC samples for accurate expectation
REGIME_WINDOW = 126                  # longer lookback for regime stress

# --- Training Epochs (aggressive) ---
DAILY_EPOCHS = 200                   # 2× previous
GLOBAL_EPOCHS = 250                  # 1.67× previous
SHRINKING_EPOCHS = 120               # 1.5× previous

# --- Training Modes ---
DAILY_LOOKBACK = 1008                # 4 years
GLOBAL_TRAIN_START = "2008-01-01"
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
