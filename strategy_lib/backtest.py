import numpy as np
import pandas as pd


def compute_portfolio_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.05) -> dict:
    returns = equity_curve.pct_change().dropna()
    n_years = len(returns) / 252

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    excess_ret = returns - risk_free_rate / 252
    sharpe = excess_ret.mean() / (excess_ret.std() + 1e-9) * np.sqrt(252)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd + 1e-9)
    hit_rate = (returns > 0).mean()

    return {
        "Total Return": f"{total_return * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Calmar Ratio": f"{calmar:.3f}",
        "Hit Rate": f"{hit_rate * 100:.2f}%",
    }


def backtest_one_ticker(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    target_hold_days: int,
    dip_threshold: float = -0.03,
):
    daily_ret = price_df["close"].pct_change().fillna(0)

    lr_pos = signal_df["signal"].shift(1).fillna(0)
    lr_returns = lr_pos * daily_ret
    lr_equity = (1 + lr_returns).cumprod()

    bh_equity = (1 + daily_ret).cumprod()

    ret_5d = price_df["close"].pct_change(5)
    in_dip = (ret_5d.shift(1) < dip_threshold).astype(float)
    dip_pos = in_dip.copy()
    for lag in range(1, target_hold_days):
        dip_pos = np.maximum(dip_pos, in_dip.shift(lag).fillna(0))

    dip_returns = dip_pos * daily_ret
    dip_equity = (1 + dip_returns).cumprod()

    equity = pd.DataFrame(
        {
            "LR Strategy": lr_equity,
            "Buy & Hold": bh_equity,
            "Buy the Dip": dip_equity,
        }
    ).dropna()

    metrics = {name: compute_portfolio_metrics(equity[name]) for name in equity.columns}
    return equity, pd.DataFrame(metrics).T
