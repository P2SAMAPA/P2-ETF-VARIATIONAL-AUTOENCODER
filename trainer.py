"""
Main training script – Daily, Global, and Shrinking modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from vae_model import VAETrainer
import push_results


def run_vae_mode(returns, macro, tickers, mode_name, epochs):
    """Train CVAE and produce forecasts for a data slice."""
    if len(returns) < config.MIN_OBSERVATIONS:
        return None

    cond, target = data_manager.build_training_sequences(returns, macro)
    if len(cond) < config.MIN_OBSERVATIONS:
        return None

    target_dim = target.shape[1]
    cond_dim = cond.shape[1]

    trainer = VAETrainer(
        target_dim=target_dim,
        cond_dim=cond_dim,
        hidden_layers=config.HIDDEN_LAYERS,
        latent_dim=config.LATENT_DIM,
        beta=config.BETA,
        lr=config.LEARNING_RATE,
        seed=config.RANDOM_SEED
    )

    print(f"  Training CVAE on {len(cond)} samples...")
    trainer.fit(cond, target, epochs=epochs, batch_size=config.BATCH_SIZE)

    # Expected returns
    latest_cond = macro.iloc[-1:].values.astype(np.float32)
    expected_returns = trainer.predict_expected_returns(
        latest_cond, tickers, num_samples=config.NUM_SAMPLES
    )

    # Regime stress
    regime_stress = trainer.compute_regime_stress(cond, target, lookback=config.REGIME_WINDOW)

    sorted_tickers = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)
    top3 = [{'ticker': t, 'expected_return': float(r)} for t, r in sorted_tickers[:3]]
    all_scores = [{'ticker': t, 'expected_return': float(r)} for t, r in sorted_tickers]

    return {
        'top_picks': top3,
        'all_scores': all_scores,
        'regime_stress': regime_stress,
        'training_start': str(returns.index[0].date()),
        'training_end': str(returns.index[-1].date()),
        'n_observations': len(returns)
    }


def run_shrinking_windows(df_master, macro, tickers, epochs):
    """Fixed shrinking windows with consensus on top ETF."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = pd.Timestamp(f"{start_year+2}-12-31")
        mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
        window_df = df_master[mask].copy()
        if len(window_df) < config.MIN_OBSERVATIONS:
            continue

        returns = data_manager.prepare_returns_matrix(window_df, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        # Align macro to returns dates (forward fill)
        m = macro.reindex(returns.index, method='ffill').dropna()
        returns_aligned = returns.loc[m.index]
        if len(returns_aligned) < config.MIN_OBSERVATIONS:
            continue

        mode_out = run_vae_mode(returns_aligned, m, tickers, f"Shrinking {start_year}", epochs)
        if mode_out:
            top_ticker = mode_out['top_picks'][0]['ticker']
            windows.append({
                'window_start': start_year,
                'window_end': start_year + 2,
                'ticker': top_ticker,
                'expected_return': mode_out['top_picks'][0]['expected_return']
            })

    if not windows:
        return None

    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {'ticker': pick, 'conviction': conviction, 'num_windows': len(windows), 'windows': windows}


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    macro = data_manager.prepare_macro_features(df_master)

    # Ensure macro is sorted for forward fill
    macro = macro.sort_index()

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue

        # --- FIX: align macro to daily returns via forward fill ---
        m = macro.reindex(returns_all.index, method='ffill').dropna()
        returns_all = returns_all.loc[m.index]   # keep only rows with macro data

        universe_out = {}

        # Daily (now uses config.DAILY_EPOCHS)
        daily_ret = returns_all.iloc[-config.DAILY_LOOKBACK:]
        daily_macro = m.iloc[-config.DAILY_LOOKBACK:]
        daily_out = run_vae_mode(daily_ret, daily_macro, tickers, "Daily",
                                 epochs=config.DAILY_EPOCHS)
        if daily_out:
            universe_out['daily'] = daily_out
            print(f"  Daily top: {daily_out['top_picks'][0]['ticker']}")

        # Global
        global_out = run_vae_mode(returns_all, m, tickers, "Global",
                                  epochs=config.GLOBAL_EPOCHS)
        if global_out:
            universe_out['global'] = global_out
            print(f"  Global top: {global_out['top_picks'][0]['ticker']}")

        # Shrinking Windows (now uses config.SHRINKING_EPOCHS)
        shrinking = run_shrinking_windows(df_master, macro, tickers,
                                          epochs=config.SHRINKING_EPOCHS)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
