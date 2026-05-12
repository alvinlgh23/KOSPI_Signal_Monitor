# KOSPI Signal Monitor

An exploratory econometric dashboard for checking candidate relationships between the KOSPI Composite (`^KS11`) and a small set of market proxies: semiconductors, USD/KRW, consumer stocks, listed K-pop agency stocks, and optional foreign-flow data.

The app uses Granger causality tests, OLS correlation/regression summaries, GARCH(1,1) volatility estimates, and wavelet signal/noise diagnostics. Outputs should be read as descriptive research signals, not proof of causality and not trading advice.

The dashboard also includes an export-earnings snapshot comparing semiconductors, ICT exports excluding semiconductors, and K-content categories such as games, music/K-pop, and broadcast/video/K-drama.

## Quick start

```bash
pip install -r requirements_v2.txt
python KOSPI_Backend_v2.py
```

Then open `http://localhost:8000`.

## Data sources

- Benchmark: Yahoo Finance `^KS11` KOSPI Composite.
- Stock proxies: yfinance Korean tickers.
- Cultural proxy: HYBE, SM Entertainment, JYP Entertainment, YG Entertainment, Cube Entertainment, FNC Entertainment, RBW, and Fantagio.
- Export earnings snapshot: MOTIR/MSIT 2025 annual exports / ICT exports, KOCCA 2025 content-industry trend totals, and latest available detailed content-category splits.
- Optional monthly goods refresh: set `DATA_GO_KR_SERVICE_KEY` or `KOREA_CUSTOMS_SERVICE_KEY` to use Korea Customs item-trade data for semiconductor HS codes `8541` and `8542`. K-content export earnings remain annual survey data because culture/content services are not published as live customs goods statistics.
- FX macro proxy: FRED `DEXKOUS`, with yfinance `KRW=X` fallback.
- Foreign-flow proxy: `pykrx`, only when KRX credentials/environment work locally.

## Accuracy notes

- Granger p-values are Bonferroni-adjusted across the tested lag window.
- Multiple sectors are screened, so results still need external validation and out-of-sample testing.
- Wavelet denoising is full-sample and descriptive; it is not safe to interpret as a real-time trading filter.
- The frontend chart plots observed indexed proxy history returned by the backend. It does not generate synthetic history.
- Scores combine adjusted p-value, correlation, and OLS R-squared into a heuristic 0-10 scale. They are useful for ranking candidates, not for estimating expected returns.
