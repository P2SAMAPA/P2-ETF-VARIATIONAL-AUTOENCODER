# P2-ETF-VARIATIONAL-AUTOENCODER

**Conditional VAE – Generative ETF Forecasting & Regime Stress Detection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-VARIATIONAL-AUTOENCODER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-VARIATIONAL-AUTOENCODER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--variational--autoencoder--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-variational-autoencoder-results)

## Overview

`P2-ETF-VARIATIONAL-AUTOENCODER` uses a **Conditional Variational Autoencoder (CVAE)** to learn the joint distribution of next‑day ETF returns conditioned on current macro variables. The latent space captures market structure, and the **KL divergence** between the encoder posterior and the prior serves as a real‑time regime stress indicator. ETFs are ranked by expected return obtained via Monte Carlo sampling from the CVAE.

## Methodology

1. **Conditioning** – Macro features (VIX, DXY, T10Y2Y, TBILL_3M) condition both encoder and decoder.
2. **CVAE Training** – Encoder compresses (next‑day return, macro) → latent z; Decoder reconstructs next‑day return from (z, macro). Loss = MSE + β·KL.
3. **Inference** – Sample 100 latent vectors from prior, decode with current macro, average to get expected returns.
4. **Regime Stress** – Average posterior KL divergence over recent lookback (63 days); high values indicate regime shift.
5. **Three Training Modes** – Daily, Global, Shrinking Windows Consensus.

## Universe

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |
