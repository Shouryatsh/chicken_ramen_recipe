import os
import re
import urllib.parse
import xml.etree.ElementTree as etree
from datetime import datetime

import numpy as np
import pandas as pd


def _ib_components():
    try:
        from ib_insync import ContFuture, IB, Index, Stock, util

        return IB, Stock, Index, ContFuture, util
    except Exception:
        return None


def get_ib_connection():
    comps = _ib_components()
    if comps is None:
        return None

    IB, *_ = comps
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "9"))

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=8)
        if ib.isConnected():
            return ib
    except Exception:
        pass
    return None


def _ib_contract_for_symbol(ticker: str):
    comps = _ib_components()
    if comps is None:
        return None

    _, Stock, Index, ContFuture, _ = comps

    symbol = ticker.upper().strip()
    if symbol == "^VIX":
        return Index("VIX", "CBOE")
    if symbol == "^TNX":
        return Index("TNX", "CBOE")
    if symbol == "DX-Y.NYB":
        return Index("DXY", "ICEUS")
    if symbol == "SPY":
        return Stock("SPY", "ARCA", "USD")
    if symbol == "^IRX":
        return Index("IRX", "NASDAQ")
    if symbol.endswith("=F"):
        root = symbol.split("=")[0]
        return ContFuture(root, "NYMEX")

    cleaned = symbol.replace(".", "-")
    return Stock(cleaned, "SMART", "USD")


def _stooq_candidates(ticker: str) -> list[str]:
    symbol = ticker.upper().strip()
    mapping = {
        "^VIX": ["^vix", "vix"],
        "^TNX": ["^tnx", "tnx"],
        "^IRX": ["^irx", "irx"],
        "DX-Y.NYB": ["dxy", "dx-y.nyb"],
        "CL=F": ["cl.f", "cl"],
        "SPY": ["spy.us", "spy"],
    }
    if symbol in mapping:
        return mapping[symbol]

    if symbol.isalpha() and len(symbol) <= 6:
        return [f"{symbol.lower()}.us", symbol.lower()]
    return [symbol.lower()]


def _fetch_stooq_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    for candidate in _stooq_candidates(ticker):
        encoded = urllib.parse.quote(candidate)
        url = f"https://stooq.com/q/d/l/?s={encoded}&i=d"
        try:
            raw = pd.read_csv(url)
        except Exception:
            continue

        if raw.empty or "Date" not in raw.columns:
            continue

        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = raw.rename(columns=rename_map)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if not df.empty:
            return df

    raise RuntimeError(f"Stooq fetch failed for {ticker}")


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    comps = _ib_components()
    if comps is not None:
        _, _, _, _, util = comps
        ib = get_ib_connection()
        if ib is not None:
            try:
                contract = _ib_contract_for_symbol(ticker)
                qualified = ib.qualifyContracts(contract)
                if qualified:
                    contract = qualified[0]

                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                duration_days = max(5, (end_dt - start_dt).days + 10)

                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=f"{duration_days} D",
                    barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                    formatDate=1,
                )
                if bars:
                    df = util.df(bars)
                    if not df.empty:
                        keep = ["date", "open", "high", "low", "close", "volume"]
                        df = df[[c for c in keep if c in df.columns]].copy()
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df.dropna(subset=["date"]).set_index("date").sort_index()
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
                        if not df.empty:
                            return df
            except Exception:
                pass
            finally:
                try:
                    ib.disconnect()
                except Exception:
                    pass

    return _fetch_stooq_ohlcv(ticker, start, end)


def _extract_ratio(xml_payload: str, keys: list[str]) -> float:
    if not xml_payload:
        return np.nan

    try:
        root = etree.fromstring(xml_payload)
    except Exception:
        return np.nan

    lowered_keys = {k.lower() for k in keys}

    for elem in root.iter():
        attrib = {str(k).lower(): str(v) for k, v in elem.attrib.items()}
        candidate = " ".join(
            [
                attrib.get("fieldname", ""),
                attrib.get("name", ""),
                attrib.get("id", ""),
                attrib.get("type", ""),
            ]
        ).lower()
        if not any(key in candidate for key in lowered_keys):
            continue

        text = (elem.text or "").strip()
        if not text:
            text = attrib.get("value", "")
        if not text:
            continue

        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group(0))
            except Exception:
                continue

    return np.nan


def fetch_fundamentals(ticker: str) -> dict:
    empty = {
        "pe_ratio": np.nan,
        "forward_pe": np.nan,
        "ps_ratio": np.nan,
        "pb_ratio": np.nan,
        "eps_ttm": np.nan,
        "eps_forward": np.nan,
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
        "profit_margin": np.nan,
        "debt_to_equity": np.nan,
        "return_on_equity": np.nan,
        "current_ratio": np.nan,
        "beta": np.nan,
    }

    ib = get_ib_connection()
    if ib is None:
        return empty

    try:
        contract = _ib_contract_for_symbol(ticker)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return empty
        contract = qualified[0]

        snapshot_xml = ib.reqFundamentalData(contract, reportType="ReportSnapshot")
        estimates_xml = ib.reqFundamentalData(contract, reportType="RESC")

        return {
            "pe_ratio": _extract_ratio(snapshot_xml, ["trailing pe", "priceearnings"]),
            "forward_pe": _extract_ratio(estimates_xml, ["forward pe", "pe fwd"]),
            "ps_ratio": _extract_ratio(snapshot_xml, ["price/sales", "ps ratio"]),
            "pb_ratio": _extract_ratio(snapshot_xml, ["price/book", "pb ratio"]),
            "eps_ttm": _extract_ratio(snapshot_xml, ["eps ttm", "eps"]),
            "eps_forward": _extract_ratio(estimates_xml, ["eps mean", "eps forward"]),
            "revenue_growth": _extract_ratio(snapshot_xml, ["revenue growth"]),
            "earnings_growth": _extract_ratio(snapshot_xml, ["earnings growth"]),
            "profit_margin": _extract_ratio(snapshot_xml, ["profit margin"]),
            "debt_to_equity": _extract_ratio(snapshot_xml, ["debt/equity", "debt to equity"]),
            "return_on_equity": _extract_ratio(snapshot_xml, ["roe", "return on equity"]),
            "current_ratio": _extract_ratio(snapshot_xml, ["current ratio"]),
            "beta": _extract_ratio(snapshot_xml, ["beta"]),
        }
    except Exception:
        return empty
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
