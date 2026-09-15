# =============================================================================
# FEATUREENGINEERING.PY — Blocco 2 · Costruzione delle feature
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from config import Config, CONFIG
from data_loader import MarketData

logger = logging.getLogger(__name__)
_MONTH_END = "ME"

# -----------------------------------------------------------------------------
# Feature engineering per singolo asset
# -----------------------------------------------------------------------------
class AssetFeatureEngineer:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.fc = config.features
        self._ann = np.sqrt(self.fc.annualization_days)

    def _mp(self, window: int, floor: int) -> int:
        # min_periods: finestra piena se richiesto, altrimenti meta' (con floor)
        return window if self.fc.require_full_window else max(floor, window // 2)

    # ---- Feature giornaliere rolling ----------------------------------------
    def compute_daily(self, close: pd.Series) -> pd.DataFrame:
        c = close.dropna().astype(float)
        if c.empty:
            return pd.DataFrame()
        r = c.pct_change()
        feat = pd.DataFrame(index=c.index)
        # Volatilità realizzate annualizzate su piu' orizzonti
        for w in self.fc.vol_windows:
            feat[f"Volatility_{w}d"] = (
                r.rolling(w, min_periods=self._mp(w, 2)).std(ddof=1) * self._ann)
        # Momenti di ordine superiore
        wm = self.fc.moment_window
        feat["Skewness_6m"] = r.rolling(wm, min_periods=self._mp(wm, 3)).skew()
        feat["Kurtosis_6m"] = r.rolling(wm, min_periods=self._mp(wm, 4)).kurt()
        # Downside deviation annualizzata
        downside_sq = r.clip(upper=0.0) ** 2
        lpm2 = downside_sq.rolling(wm, min_periods=self._mp(wm, 2)).mean()
        feat["DownsideVol_6m"] = np.sqrt(lpm2) * self._ann
        # Max Drawdown rolling
        for label, w in self.fc.drawdown_windows.items():
            roll_max = c.rolling(w, min_periods=self._mp(w, 2)).max()
            feat[f"Drawdown_{label}"] = c / roll_max - 1.0
        return feat

    # ---- Feature mensili su prezzi di fine mese -----------------------------
    def compute_monthly_price_features(self, close: pd.Series) -> pd.DataFrame:
        # Reversal a 1 mese e momentum a piu' orizzonti
        p = close.dropna().resample(_MONTH_END).last()
        out = pd.DataFrame(index=p.index)
        k = self.fc.reversal_months
        out["Reversal_1m"] = p / p.shift(k) - 1.0
        for label, near, far in self.fc.momentum_specs:
            out[label] = p.shift(near) / p.shift(far) - 1.0
        out.index.name = "date"
        return out

# -----------------------------------------------------------------------------
# Feature engineering a livello di framework
# -----------------------------------------------------------------------------
@dataclass
class FeatureSet:
    asset_daily: Dict[str, pd.DataFrame]   # feature giornaliere per ticker
    asset_monthly: pd.DataFrame            # pannello mensile (date, ticker)
    macro_daily: pd.DataFrame              # feature macro point-in-time
    macro_monthly: pd.DataFrame            # macro a fine mese
    forward_returns: pd.DataFrame          # target: rendimenti forward a 1 mese

class FeatureEngineer:
    # Costruisce tutte le feature del framework a partire dai dati di mercato
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.fc = config.features
        self._asset_eng = AssetFeatureEngineer(config)

    @property
    def feature_names(self) -> List[str]:
        # Ordine canonico delle feature di asset (colonne del LASSO)
        names = ["Reversal_1m"]
        names += [label for label, _, _ in self.fc.momentum_specs]
        names += [f"Volatility_{w}d" for w in self.fc.vol_windows]
        names += ["Skewness_6m", "Kurtosis_6m", "DownsideVol_6m"]
        names += [f"Drawdown_{label}" for label in self.fc.drawdown_windows]
        return names

    # ---- Feature rolling di asset -------------------------------------------
    def asset_features_daily(self, market: MarketData) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for ticker in self.cfg.tickers:
            if ticker not in market.prices_eur.columns:
                continue
            out[ticker] = self._asset_eng.compute_daily(market.prices_eur[ticker])
        return out

    def asset_features_monthly(self, market: MarketData,
                               asset_daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Costruisce il pannello mensile (date, ticker)
        frames = []
        order = self.feature_names
        for ticker in self.cfg.tickers:
            close = market.prices_eur.get(ticker)
            if close is None or close.dropna().empty:
                continue
            daily = asset_daily.get(ticker)
            m_roll = (daily.resample(_MONTH_END).last()
                      if daily is not None and not daily.empty else pd.DataFrame())
            m_price = self._asset_eng.compute_monthly_price_features(close)
            monthly = m_price.join(m_roll, how="outer")
            # Aggiunge eventuali colonne mancanti e impone l'ordine canonico
            for col in order:
                if col not in monthly.columns:
                    monthly[col] = np.nan
            monthly = monthly[order].dropna(how="all")
            if monthly.empty:
                continue
            monthly.index.name = "date"
            monthly["ticker"] = ticker
            frames.append(monthly.set_index("ticker", append=True))
            logger.info("Feature mensili %s: %d mesi.", ticker, len(monthly))
        if not frames:
            return pd.DataFrame(columns=order)
        return pd.concat(frames).sort_index()

    # ---- Forward returns (target) -------------------------------------------
    def forward_returns(self, market: MarketData) -> pd.DataFrame:
        h = self.cfg.lasso.forecast_horizon_months
        monthly_px = market.prices_eur.resample(_MONTH_END).last()
        fwd = np.log(monthly_px.shift(-h) / monthly_px)
        fwd.index.name = "date"
        return fwd

    # ---- Feature macro giornaliere point-in-time ----------------------------
    def macro_features_daily(self, market: MarketData) -> pd.DataFrame:
        # Livello e variazione di ogni serie macro
        primary = {
            "VIX": "CBOE Volatility Index",
            "HY_OAS": "High Yield Index Option-Adjusted Spread",
            "NFCI": "National Financial Conditions Index",
            "SLOPE": "SLOPE 10Y Treasury 2Y Treasury",
            "REAL_RATE": "REAL RATE",
            "BREAKEVEN": "10-Year Breakeven Inflation",
            "DXY": "US Dollar Index",
            "PPI": "Producer Price Index by Commodity",
        }
        cols: Dict[str, pd.Series] = {}
        for name, df in market.macro.items():
            col = primary.get(name)
            if col is None or col not in df.columns:
                num = df.select_dtypes("number")
                if num.empty:
                    continue
                col = num.columns[0]
            level = pd.to_numeric(df[col], errors="coerce")
            cols[f"{name}_level"] = level
            cols[f"{name}_chg"] = level.diff()
        macro = pd.DataFrame(cols).sort_index()
        idx = market.prices_eur.index
        macro = macro.reindex(macro.index.union(idx)).ffill(
            limit=self.cfg.data.max_ffill_days).reindex(idx)
        return macro.shift(1)   # point-in-time: info nota fino a t-1

    def macro_features_monthly(self, macro_daily: pd.DataFrame) -> pd.DataFrame:
        # Campiona le macro a fine mese
        return macro_daily.resample(_MONTH_END).last()

    # ---- Orchestrazione completa --------------------------------------------
    def build(self, market: MarketData) -> FeatureSet:
        asset_daily = self.asset_features_daily(market)
        asset_monthly = self.asset_features_monthly(market, asset_daily)
        macro_daily = self.macro_features_daily(market)
        macro_monthly = self.macro_features_monthly(macro_daily)
        fwd = self.forward_returns(market)
        n_months = (asset_monthly.index.get_level_values("date").nunique()
                    if not asset_monthly.empty else 0)
        logger.info("Feature engineering: %d asset, %d mesi, %d feature macro.",
                    len(asset_daily), n_months, macro_monthly.shape[1])
        return FeatureSet(asset_daily=asset_daily, asset_monthly=asset_monthly,
                          macro_daily=macro_daily, macro_monthly=macro_monthly,
                          forward_returns=fwd)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data_loader import DataLoader
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    CONFIG.data.data_dir = Path(args.data_dir)
    CONFIG.data.fred_dir = Path(args.data_dir)
    md = DataLoader(CONFIG).load()
    fe = FeatureEngineer(CONFIG)
    fs = fe.build(md)
    print("Feature di asset:", fe.feature_names)
    print("Panel mensile:", fs.asset_monthly.shape, "| mesi:",
          fs.asset_monthly.index.get_level_values("date").nunique())
    last = fs.asset_monthly.index.get_level_values("date").max()
    print(f"\nCross-section a {last.date()}:")
    print(fs.asset_monthly.xs(last, level="date").round(4).to_string())
    print("\nFeature macro mensili:", fs.macro_monthly.shape, list(fs.macro_monthly.columns)[:6], "...")