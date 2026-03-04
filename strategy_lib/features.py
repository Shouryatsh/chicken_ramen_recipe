import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .data import _ib_contract_for_symbol, fetch_ohlcv, get_ib_connection


_analyzer = SentimentIntensityAnalyzer()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close, window=26).ema_indicator()

    df["price_vs_sma20"] = (close - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"]
    df["ema_crossover"] = df["ema_12"] - df["ema_26"]

    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    df["roc_5"] = ROCIndicator(close, window=5).roc()
    df["roc_10"] = ROCIndicator(close, window=10).roc()

    macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_sig"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["bb_pct"] = bb.bollinger_pband()

    atr_obj = AverageTrueRange(high, low, close, window=14)
    df["atr_14"] = atr_obj.average_true_range()
    df["atr_pct"] = df["atr_14"] / close

    df["vol_ratio"] = volume / volume.rolling(20).mean()
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv_change"] = obv.pct_change(5)

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["rvol_10d"] = df["ret_1d"].rolling(10).std() * np.sqrt(252)
    return df


def _score_headline(text: str) -> float:
    return _analyzer.polarity_scores(str(text))["compound"]


def fetch_ibkr_news_sentiment(ticker: str, n_articles: int = 30) -> float:
    ib = None
    try:
        ib = get_ib_connection()
        if ib is None:
            return 0.0

        contract = _ib_contract_for_symbol(ticker)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return 0.0

        con_id = qualified[0].conId
        providers = ib.reqNewsProviders()
        provider_codes = "+".join([provider.code for provider in providers if provider.code][:8])
        if not provider_codes:
            return 0.0

        news = ib.reqHistoricalNews(
            conId=con_id,
            providerCodes=provider_codes,
            startDateTime="",
            endDateTime="",
            totalResults=n_articles,
        )

        scores = []
        for article in news:
            title = getattr(article, "headline", "") or ""
            if title:
                scores.append(_score_headline(title))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def add_sentiment_features(df: pd.DataFrame, live_sentiment: float, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    ret5 = df["ret_5d"].fillna(0)
    sentiment_series = (
        0.4 * np.sign(ret5.shift(1)).fillna(0)
        + 0.6 * np.random.normal(0, 0.15, len(df))
    )
    sentiment_series = sentiment_series.clip(-1, 1)

    df["sentiment_raw"] = sentiment_series
    df["sentiment_ema3"] = sentiment_series.ewm(span=3).mean()
    df["sentiment_mom5"] = df["sentiment_ema3"].diff(5)

    df.loc[df.index[-1], "sentiment_raw"] = live_sentiment
    df.loc[df.index[-1], "sentiment_ema3"] = live_sentiment
    return df


def add_momentum_features(df: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    close = df["close"]
    df["mom_1m"] = close.pct_change(21)
    df["mom_3m"] = close.pct_change(63)
    df["mom_6m"] = close.pct_change(126)
    df["mom_12m"] = close.pct_change(252)

    high_52w = close.rolling(252).max()
    low_52w = close.rolling(252).min()
    df["pct_from_52w_high"] = (close - high_52w) / high_52w
    df["pct_from_52w_low"] = (close - low_52w) / low_52w
    df["range_position_52w"] = (close - low_52w) / (high_52w - low_52w + 1e-9)

    def fip_score(rets: pd.Series, window: int = 21):
        sign_changes = (np.sign(rets) != np.sign(rets.shift(1))).rolling(window).mean()
        return 1 - sign_changes

    df["fip_21d"] = fip_score(df["ret_1d"], 21)
    spy_ret = spy.pct_change(21).reindex(df.index).ffill()
    df["rs_vs_spy_1m"] = df["mom_1m"] - spy_ret
    df["mom_accel"] = df["mom_1m"] - df["mom_1m"].shift(5)
    return df


def build_macro_enhanced(macro_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        tnx_2y_raw = fetch_ohlcv("^IRX", start_date, end_date)
        tnx_2y = tnx_2y_raw[["close"]].rename(columns={"close": "IRX"})
    except Exception:
        tnx_2y = pd.DataFrame({"IRX": np.nan}, index=macro_df.index)

    macro_enhanced = macro_df.copy().join(tnx_2y, how="left").ffill()
    macro_enhanced["vix_chg_5d"] = macro_enhanced["VIX"].pct_change(5)
    macro_enhanced["vix_zscore"] = (
        (macro_enhanced["VIX"] - macro_enhanced["VIX"].rolling(63).mean())
        / macro_enhanced["VIX"].rolling(63).std()
    )
    macro_enhanced["dxy_chg_5d"] = macro_enhanced["DXY"].pct_change(5)
    macro_enhanced["tnx_chg_5d"] = macro_enhanced["TNX"].pct_change(5)
    macro_enhanced["oil_chg_5d"] = macro_enhanced["OIL"].pct_change(5)
    macro_enhanced["yield_slope"] = macro_enhanced["TNX"] - macro_enhanced.get(
        "IRX", pd.Series(0, index=macro_enhanced.index)
    )
    macro_enhanced["spy_trend"] = (
        macro_enhanced["SPY"] / macro_enhanced["SPY"].rolling(50).mean()
    ) - 1
    macro_enhanced["regime_risk_on"] = (
        (macro_enhanced["VIX"] < 20) & (macro_enhanced["spy_trend"] > 0)
    ).astype(int)
    return macro_enhanced
