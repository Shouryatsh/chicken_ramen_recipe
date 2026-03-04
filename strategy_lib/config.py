from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class StrategyConfig:
    stocks: list[str]
    hold_periods: dict[str, int]
    target_hold: str
    threshold: float
    start_date: str
    end_date: str
    macro_tickers: dict[str, str]


FEATURE_COLS = [
    "rsi_14", "macd", "macd_sig", "macd_hist",
    "bb_width", "bb_pct", "atr_pct",
    "price_vs_sma20", "price_vs_sma50", "ema_crossover",
    "roc_5", "roc_10", "vol_ratio", "obv_change", "rvol_10d",
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "fip_21d", "rs_vs_spy_1m", "mom_accel",
    "pct_from_52w_high", "pct_from_52w_low", "range_position_52w",
    "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
    "revenue_growth", "earnings_growth", "profit_margin",
    "debt_to_equity", "return_on_equity", "beta",
    "sentiment_raw", "sentiment_ema3", "sentiment_mom5",
    "VIX", "vix_chg_5d", "vix_zscore",
    "DXY", "dxy_chg_5d",
    "TNX", "tnx_chg_5d",
    "oil_chg_5d", "yield_slope", "spy_trend",
    "regime_risk_on",
]


def default_config() -> StrategyConfig:
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    return StrategyConfig(
        stocks=["NVDA", "MSFT"],
        hold_periods={"1W": 5, "2W": 10, "1M": 21},
        target_hold="1W",
        threshold=0.0025,
        start_date=start_date,
        end_date=end_date,
        macro_tickers={
            "VIX": "^VIX",
            "DXY": "DX-Y.NYB",
            "TNX": "^TNX",
            "OIL": "CL=F",
            "SPY": "SPY",
        },
    )
