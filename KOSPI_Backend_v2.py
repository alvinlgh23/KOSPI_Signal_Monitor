"""
KOSPI Multi-Factor Signal Monitor v2.1
Pure econometrics — no AI API needed.
Enhanced: richer strength labels, negative bias detection, regime detection,
          action layer, signal ranking, narrative interpretation.
"""

import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from arch import arch_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except ImportError:
    HAS_FDR = False

try:
    from pykrx import stock as krx
    HAS_KRX = True
except ImportError:
    HAS_KRX = False

try:
    import pandas_datareader.data as web
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

app = FastAPI(title="KOSPI Signal Monitor v2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BENCHMARK = "^KS200"
TIMEFRAME_DAYS = {"3m": 90, "6m": 180, "1y": 365, "2y": 730}
FRED_SERIES = {
    "usd_krw": "DEXKOUS",
    "us_fed_rate": "FEDFUNDS",
    "korea_manufacturing": "KORPROMAN02IXOBSAM",
}


# ── Pydantic models ───────────────────────────────────────────
class SweepRequest(BaseModel):
    sectors: list[str]
    timeframe: str = "1y"

class SignalResult(BaseModel):
    key: str
    label: str
    score: float
    corr: float
    pval: float
    strength: str
    strength_short: str
    data_source: str
    detail: dict

class SweepResponse(BaseModel):
    results: list[SignalResult]
    volatility: dict
    interpretation: dict
    timestamp: str
    timeframe: str


# ── Strength label logic ──────────────────────────────────────
def strength_label(score: float, corr: float, pval: float) -> tuple[str, str]:
    """
    Returns (full_label, short_label).
    Incorporates direction (positive/negative) and significance.
    """
    direction = "Positive" if corr >= 0 else "Negative"
    bias = "Bias" if corr < 0 else ""

    if score > 7:
        if corr < 0:
            return f"Strong ({direction} Bias)", "STRONG–"
        return "Strong", "STRONG"
    elif score > 4:
        if corr < 0:
            return f"Moderate ({direction} Bias)", "MOD–"
        return "Moderate", "MOD"
    elif score > 2:
        if corr < 0:
            return f"Weak ({direction} Bias)", "WEAK–"
        return "Weak", "WEAK"
    elif pval < 0.15 and abs(corr) > 0.05:
        if corr < 0:
            return f"Marginal ({direction} Bias)", "MARG–"
        return "Marginal", "MARG"
    else:
        return "No Signal", "NONE"


# ── Data fetchers ─────────────────────────────────────────────
def fetch_benchmark(days: int) -> pd.Series:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = yf.download(BENCHMARK, start=start, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.dropna()

def fetch_fred_series(series_id: str, days: int) -> Optional[pd.Series]:
    if not HAS_FRED:
        return None
    try:
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = web.DataReader(series_id, "fred", start)
        s = df.iloc[:, 0].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.resample("D").interpolate().ffill()
    except Exception as e:
        logger.warning(f"FRED fetch failed ({series_id}): {e}")
        return None

def fetch_stock_proxy(tickers: list[str], days: int) -> Optional[pd.Series]:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    series_list = []
    for t in tickers:
        try:
            data = yf.download(t, start=start, progress=False)["Close"]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            data.index = pd.to_datetime(data.index).tz_localize(None)
            normalized = data / data.iloc[0]
            series_list.append(normalized.dropna())
        except Exception as e:
            logger.warning(f"Stock fetch failed ({t}): {e}")
    if not series_list:
        return None
    combined = pd.concat(series_list, axis=1).mean(axis=1)
    return combined.resample("D").interpolate().ffill()

def fetch_foreign_flows(days: int) -> Optional[pd.Series]:
    if not HAS_KRX:
        return None
    try:
        end = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
        df = krx.get_market_trading_value_by_date(start, end, "KOSPI")
        if df.empty:
            return None
        foreign_col = [c for c in df.columns if "외국인" in str(c) or "Foreign" in str(c)]
        col = foreign_col[0] if foreign_col else df.columns[0]
        s = df[col].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.resample("D").interpolate().ffill()
    except Exception as e:
        logger.warning(f"pykrx foreign flow failed: {e}")
        return None

def get_sector_signal(sector_key: str, days: int) -> tuple[Optional[pd.Series], str]:
    if sector_key == "cultural":
        s = fetch_stock_proxy(["352820.KS", "041510.KS"], days)
        return s, "HYBE + SM Entertainment (normalized price proxy)"
    elif sector_key == "tech":
        s = fetch_stock_proxy(["000660.KS", "005930.KS"], days)
        return s, "SK Hynix + Samsung Electronics (normalized price proxy)"
    elif sector_key == "macro":
        s = fetch_fred_series(FRED_SERIES["usd_krw"], days)
        if s is not None and len(s) > 30:
            return s, "USD/KRW exchange rate (FRED)"
        try:
            start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = yf.download("KRW=X", start=start, progress=False)["Close"]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            data.index = pd.to_datetime(data.index).tz_localize(None)
            return data.dropna(), "USD/KRW exchange rate (yfinance fallback)"
        except Exception:
            return None, "unavailable"
    elif sector_key == "consumption":
        # Broad domestic/global consumer demand proxy:
        # K-beauty:     Amorepacific (090430.KS), LG H&H (051900.KS)
        # Consumer tech/electronics: Samsung SDI (006400.KS), LG Electronics (066570.KS)
        # Automotive:   Hyundai Motor (005380.KS), Kia (000270.KS)
        # Retail/FMCG:  BGF Retail (282330.KS), GS Retail (007070.KS)
        tickers = [
            "090430.KS",  # Amorepacific   — K-beauty / cosmetics
            "051900.KS",  # LG H&H         — beauty + household
            "066570.KS",  # LG Electronics — consumer electronics, TVs, appliances
            "005380.KS",  # Hyundai Motor  — automotive
            "000270.KS",  # Kia            — automotive
            "282330.KS",  # BGF Retail     — convenience retail
        ]
        s = fetch_stock_proxy(tickers, days)
        return s, "Broad consumption basket: K-beauty · Electronics · Auto · Retail (6 stocks, normalized)"
    elif sector_key == "foreign_flow":
        s = fetch_foreign_flows(days)
        return s, "Net foreign investor buying on KOSPI (pykrx)"
    return None, "unknown sector"


# ── Signal processing ─────────────────────────────────────────
def wavelet_decompose(series: pd.Series, wavelet: str = "db4", level: int = 3):
    values = series.dropna().values
    if len(values) < 2 ** (level + 1):
        return series, pd.Series(0, index=series.index)
    coeffs = pywt.wavedec(values, wavelet, level=level)
    coeffs_trend = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend_values = pywt.waverec(coeffs_trend, wavelet)[:len(values)]
    trend = pd.Series(trend_values, index=series.dropna().index)
    noise = series.dropna() - trend
    return trend, noise

def compute_garch_volatility(returns: pd.Series) -> dict:
    clean = returns.dropna() * 100
    if len(clean) < 50:
        return {"current_vol": None, "regime": "Insufficient Data", "clustering": False}
    try:
        model = arch_model(clean, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = model.fit(disp="off", show_warning=False)
        cond_vol = res.conditional_volatility
        current_vol = float(cond_vol.iloc[-1]) / 100
        avg_vol = float(cond_vol.mean()) / 100
        clustering = current_vol > avg_vol * 1.3
        if current_vol < 0.008: regime = "Low"
        elif current_vol < 0.015: regime = "Normal"
        elif current_vol < 0.025: regime = "Elevated"
        else: regime = "High"
        alpha = float(res.params.get("alpha[1]", 0))
        beta = float(res.params.get("beta[1]", 0))
        return {
            "current_vol": round(current_vol * 100, 3),
            "avg_vol": round(avg_vol * 100, 3),
            "regime": regime,
            "clustering": clustering,
            "persistence": round(alpha + beta, 4),
        }
    except Exception as e:
        logger.warning(f"GARCH failed: {e}")
        return {"current_vol": None, "regime": "Model Error", "clustering": False}

def compute_signal_score(signal_series: pd.Series, price_series: pd.Series) -> tuple[float, float, float, dict]:
    combined = pd.DataFrame({"price": price_series, "signal": signal_series}).dropna()
    if len(combined) < 30:
        return 0.0, 0.0, 1.0, {"error": "insufficient data"}

    price_ret = np.log(combined["price"] / combined["price"].shift(1))
    signal_ret = np.log(combined["signal"].clip(lower=1e-9) / combined["signal"].clip(lower=1e-9).shift(1))
    returns = pd.DataFrame({"price": price_ret, "signal": signal_ret})
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 20:
        return 0.0, 0.0, 1.0, {"error": "insufficient returns"}

    wavelet_snr = None
    if len(returns) >= 32:
        try:
            trend, noise = wavelet_decompose(returns["signal"])
            signal_power = float(returns["signal"].var())
            noise_power = float(noise.var()) if noise.var() > 0 else 1e-9
            wavelet_snr = round(signal_power / noise_power, 3)
            returns = returns.copy()
            returns["signal"] = trend.reindex(returns.index).ffill()
            returns = returns.dropna()
        except Exception:
            pass

    corr = float(returns["price"].corr(returns["signal"]))

    try:
        gc_res = grangercausalitytests(returns[["price", "signal"]], maxlag=7, verbose=False)
        p_values = [gc_res[lag + 1][0]["ssr_ftest"][1] for lag in range(7)]
        min_p = float(min(p_values))
        best_lag = int(np.argmin(p_values)) + 1
    except Exception:
        min_p = 1.0
        best_lag = 0

    try:
        X = add_constant(returns["signal"])
        ols_res = OLS(returns["price"], X).fit()
        beta = float(ols_res.params.get("signal", 0))
        r_squared = float(ols_res.rsquared)
    except Exception:
        beta = 0.0
        r_squared = 0.0

    score = 0.0
    if min_p < 0.01: score += 5.0
    elif min_p < 0.05: score += 3.5
    elif min_p < 0.10: score += 2.0

    if abs(corr) > 0.3: score += 3.0
    elif abs(corr) > 0.15: score += 2.0
    elif abs(corr) > 0.05: score += 1.0

    if r_squared > 0.1: score += 2.0
    elif r_squared > 0.05: score += 1.0

    score = score * (1.0 - min_p)

    detail = {
        "best_lag": best_lag,
        "beta": round(beta, 4),
        "r_squared": round(r_squared, 4),
        "wavelet_snr": wavelet_snr,
        "obs": len(returns),
        "direction": "positive" if corr >= 0 else "negative",
    }
    return round(score, 4), round(corr, 4), round(min_p, 4), detail


# ── Regime detection ──────────────────────────────────────────
def detect_regime(results: list[dict], vol_data: dict) -> dict:
    """
    Classify the current market into a named regime based on which
    signals are statistically dominant.
    """
    strong = [r for r in results if r["score"] > 7]
    moderate = [r for r in results if 4 < r["score"] <= 7]
    top = sorted(results, key=lambda x: x["score"], reverse=True)

    vol_regime = vol_data.get("regime", "Normal")
    clustering = vol_data.get("clustering", False)
    persistence = vol_data.get("persistence", 0)

    # Primary regime name
    if not top or top[0]["score"] < 2:
        regime_name = "Indeterminate"
        regime_desc = "No sector shows statistically meaningful explanatory power."
    else:
        top_key = top[0]["key"]
        top_dir = top[0]["detail"].get("direction", "positive")

        if top_key == "tech":
            regime_name = "Tech-Driven"
            regime_desc = "Semiconductor/AI sector momentum is the primary KOSPI driver."
        elif top_key == "macro":
            if top_dir == "negative":
                regime_name = "Macro-Driven (FX Stress)"
                regime_desc = "KRW weakness / USD strength is suppressing KOSPI returns."
            else:
                regime_name = "Macro-Driven"
                regime_desc = "Macro conditions (FX, rates) are the primary KOSPI driver."
        elif top_key == "cultural":
            regime_name = "Sentiment-Driven"
            regime_desc = "K-wave / cultural sector momentum is leading KOSPI moves."
        elif top_key == "consumption":
            regime_name = "Consumption-Driven"
            regime_desc = "Broad consumer demand (autos, electronics, beauty, retail) is the primary KOSPI driver."
        elif top_key == "foreign_flow":
            regime_name = "Flow-Driven"
            regime_desc = "Foreign investor net buying/selling is the dominant force."
        else:
            regime_name = f"{top[0]['label']}-Driven"
            regime_desc = f"{top[0]['label']} is the primary market driver."

    # Volatility overlay
    if vol_regime in ["High", "Elevated"] and clustering:
        vol_tag = "High-Vol Clustering"
    elif vol_regime == "High":
        vol_tag = "High Volatility"
    elif vol_regime == "Low":
        vol_tag = "Low Volatility"
    else:
        vol_tag = f"{vol_regime} Volatility"

    return {
        "name": regime_name,
        "description": regime_desc,
        "vol_tag": vol_tag,
        "dominant_count": len(strong),
        "moderate_count": len(moderate),
    }


# ── Interpretation engine ─────────────────────────────────────
def generate_interpretation(results: list[dict], vol_data: dict) -> dict:
    sorted_r = sorted(results, key=lambda x: x["score"], reverse=True)
    top = sorted_r[0] if sorted_r else None
    regime = detect_regime(results, vol_data)

    vol_regime = vol_data.get("regime", "Unknown")
    clustering = vol_data.get("clustering", False)
    persistence = vol_data.get("persistence", 0)
    current_vol = vol_data.get("current_vol")

    sentences = []
    action_lines = []

    # ── Primary driver sentence ──
    if top and top["score"] > 2:
        direction_phrase = (
            "positively correlated" if top["detail"].get("direction") == "positive"
            else "negatively correlated (inverse relationship)"
        )
        primary_sentence = (
            f"{top['label']} is the dominant market driver "
            f"(score {top['score']:.2f}/10, p={top['pval']:.3f}), "
            f"{direction_phrase} with ^KS200 returns "
            f"(r={top['corr']:.3f}, β={top['detail'].get('beta', 0):.4f}). "
            f"Best predictive lag: {top['detail'].get('best_lag', '?')} trading day(s)."
        )
        sentences.append(primary_sentence)

        # Add sector-specific context line
        if top["key"] == "consumption":
            sentences.append(
                "Consumption proxy covers broad domestic and export-driven consumer demand: "
                "K-beauty (Amorepacific, LG H&H), consumer electronics (LG Electronics), "
                "automotive (Hyundai, Kia), and retail (BGF Retail). "
                "A rising signal indicates broad-based consumer sector strength."
            )
        elif top["key"] == "tech":
            sentences.append(
                "Tech proxy tracks semiconductor and electronics manufacturing momentum via "
                "SK Hynix and Samsung Electronics — Korea's two largest chip exporters and primary KOSPI weights."
            )
        elif top["key"] == "cultural":
            sentences.append(
                "Cultural proxy tracks K-wave commercial momentum via HYBE and SM Entertainment — "
                "leading indicators of Hallyu export strength and soft-power sentiment."
            )
        elif top["key"] == "macro":
            sentences.append(
                "Macro proxy tracks USD/KRW — a stronger dollar (higher value) typically signals "
                "headwinds for KOSPI as it raises import costs and pressures export margins."
            )
    else:
        sentences.append(
            "No sector shows statistically significant Granger causality with ^KS200 "
            f"over this timeframe (best score: {top['score']:.2f}/10 — below significance threshold). "
            "The market appears driven by global macro or idiosyncratic factors not captured in this model."
        )

    # ── Secondary / no-signal sectors ──
    no_signal = [r for r in sorted_r if r["score"] <= 2]
    moderate_sigs = [r for r in sorted_r if 2 < r["score"] <= 7 and r != top]

    if no_signal:
        names = ", ".join(r["label"] for r in no_signal)
        sentences.append(
            f"No reliable explanatory power detected in: {names}. "
            "These sectors show p-values above significance thresholds and negligible correlation with index returns."
        )

    if moderate_sigs:
        for r in moderate_sigs:
            dir_word = "positive" if r["detail"].get("direction") == "positive" else "negative"
            sentences.append(
                f"{r['label']} shows a {r['strength'].lower()} signal "
                f"(score {r['score']:.2f}, r={r['corr']:.3f}, p={r['pval']:.3f}) "
                f"with a {dir_word} directional bias."
            )

    # ── Volatility context ──
    if current_vol is not None:
        vol_sentence = f"KOSPI volatility regime: {vol_regime} ({current_vol:.2f}%/day, GARCH(1,1))"
        if clustering:
            vol_sentence += " — volatility clustering detected, suggesting recent moves may persist"
        if persistence and persistence > 0.85:
            vol_sentence += f"; high shock persistence ({persistence:.3f})"
        sentences.append(vol_sentence + ".")

    # ── Wavelet note ──
    if top and top["detail"].get("wavelet_snr") is not None:
        snr = top["detail"]["wavelet_snr"]
        if snr > 5:
            sentences.append(
                f"Wavelet decomposition confirms a strong trend component in {top['label']} "
                f"(SNR={snr:.1f}x) — low noise contamination in the Granger test."
            )
        elif snr < 2:
            sentences.append(
                f"Note: {top['label']} signal is noise-dominated (wavelet SNR={snr:.1f}x). "
                "Score may be overstated — interpret with caution."
            )

    # ── Action layer ──
    if top and top["score"] > 7:
        action_lines.append(
            f"Market is likely sensitive to {top['label']}-related news and earnings. "
            f"Directional bias: {'bullish' if top['detail'].get('direction') == 'positive' else 'bearish'} on {top['label']} strength."
        )
        action_lines.append(
            f"Monitor {top['label']} sector catalysts with a {top['detail'].get('best_lag', '?')}-day lead for KOSPI positioning."
        )
    elif top and top["score"] > 4:
        action_lines.append(
            f"Moderate signal environment — {top['label']} has partial explanatory power but confirmation is needed before directional exposure."
        )
    else:
        action_lines.append(
            "Weak signal environment. Watch for macro event risk (Fed decisions, KRX data releases, earnings) as near-term catalysts."
        )

    if vol_regime in ["High", "Elevated"] and clustering:
        action_lines.append(
            "High volatility with clustering — consider wider position sizing and tighter stop-losses."
        )

    # ── Signal ranking ──
    ranking = []
    SECTOR_DEFS = {
        "cultural":    {"proxy": "HYBE + SM Entertainment", "scope": "K-wave / cultural export momentum"},
        "tech":        {"proxy": "SK Hynix + Samsung Electronics", "scope": "Semiconductor / AI chip demand"},
        "macro":       {"proxy": "USD/KRW exchange rate", "scope": "FX stress / macro conditions"},
        "consumption": {"proxy": "Amorepacific · LG H&H · LG Electronics · Hyundai · Kia · BGF Retail", "scope": "Domestic + global consumer demand (beauty · auto · electronics · retail)"},
        "foreign_flow":{"proxy": "Net foreign KOSPI buy/sell (pykrx)", "scope": "Foreign investor flow direction"},
    }
    for i, r in enumerate(sorted_r, 1):
        defn = SECTOR_DEFS.get(r["key"], {})
        ranking.append({
            "rank": i,
            "label": r["label"],
            "score": r["score"],
            "strength": r["strength"],
            "direction": r["detail"].get("direction", "—"),
            "proxy": defn.get("proxy", "—"),
            "scope": defn.get("scope", "—"),
        })

    return {
        "sentences": sentences,
        "action": action_lines,
        "ranking": ranking,
        "regime": regime,
        "primary_driver": top["label"] if top else "—",
    }


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "Frontend" / "dashboard_v4.html",
        base_dir / "frontend" / "dashboard_v4.html",
        base_dir / "Frontend" / "dashboard_v3.html",
        base_dir / "frontend" / "dashboard_v3.html",
        base_dir / "Frontend" / "index.html",
        base_dir / "frontend" / "index.html",
    ]
    for path in candidates:
        if path.exists():
            return FileResponse(path)
    return {"error": "frontend not found — place dashboard_v4.html in Frontend/ or frontend/"}

@app.get("/health")
def health():
    return {"status": "ok", "fdr": HAS_FDR, "krx": HAS_KRX, "fred": HAS_FRED, "gemini": False}

@app.post("/sweep", response_model=SweepResponse)
def run_sweep(req: SweepRequest):
    if not req.sectors:
        raise HTTPException(status_code=400, detail="No sectors specified.")
    days = TIMEFRAME_DAYS.get(req.timeframe, 365)
    logger.info(f"Sweep — sectors={req.sectors}, days={days}")

    try:
        market_data = fetch_benchmark(days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Benchmark fetch failed: {e}")

    bench_returns = np.log(market_data / market_data.shift(1)).dropna()
    vol_data = compute_garch_volatility(bench_returns)

    label_map = {
        "cultural": "Cultural", "tech": "Tech / AI",
        "macro": "Macro", "consumption": "Consumption", "foreign_flow": "Foreign Flows",
    }

    results_raw = []
    for sector_key in req.sectors:
        signal, source = get_sector_signal(sector_key, days)
        label = label_map.get(sector_key, sector_key.title())

        if signal is None or len(signal.dropna()) < 20:
            results_raw.append({
                "key": sector_key, "label": label,
                "score": 0.0, "corr": 0.0, "pval": 1.0,
                "strength": "No Signal", "strength_short": "NONE",
                "data_source": source or "unavailable",
                "detail": {"error": "no data", "direction": "—", "best_lag": 0, "beta": 0, "r_squared": 0, "wavelet_snr": None, "obs": 0},
            })
            continue

        score, corr, pval, detail = compute_signal_score(signal, market_data)
        full_label, short_label = strength_label(score, corr, pval)

        results_raw.append({
            "key": sector_key, "label": label,
            "score": score, "corr": corr, "pval": pval,
            "strength": full_label, "strength_short": short_label,
            "data_source": source, "detail": detail,
        })

    interpretation = generate_interpretation(results_raw, vol_data)

    return SweepResponse(
        results=[SignalResult(**r) for r in results_raw],
        volatility=vol_data,
        interpretation=interpretation,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        timeframe=req.timeframe,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("KOSPI_Backend_v2:app", host="0.0.0.0", port=8000, reload=True)
