# =============================================================================
# REGIMEDETECTION.PY — Blocco 5 · Rilevamento dei regimi via Random Forest
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import Config, CONFIG
from data_loader import MarketData
from feature_engineering import FeatureSet

logger = logging.getLogger(__name__)
_MONTH_END = "ME"
_TRADING_DAYS = 252
_PCT_MIN_PERIODS = 24        # osservazioni minime per una soglia a percentile sensata

# ---------------------------------------------------------------------------
# Contenitore di output
# ---------------------------------------------------------------------------
@dataclass
class RegimeBook:
    regimes: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))       # 'normal'/'stress'
    prob_stress: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))    # probabilità lisciata
    prob_raw: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))       # probabilità grezza
    labels: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))         # etichetta di stress (target)
    triggers: pd.DataFrame = field(default_factory=pd.DataFrame)                      # flag dei 4 trigger
    importances: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))    # importanza media feature
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def regime_at(self, date: pd.Timestamp, default: str = "normal") -> str:
        # Regime previsto per il mese richiesto (default se assente)
        if date in self.regimes.index and isinstance(self.regimes.loc[date], str):
            return self.regimes.loc[date]
        return default

# ---------------------------------------------------------------------------
# Rilevatore di regimi
# ---------------------------------------------------------------------------
class RegimeDetector:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.rc = config.regime
        self.n_triggers_stress = int(self.rc.n_triggers_stress)
        self.feature_lag = int(self.rc.feature_lag_months)
        self.stress_threshold = float(self.rc.stress_prob_threshold)
        self.trigger_mode = str(getattr(self.rc, "trigger_mode", "level"))
        self.delta_h = int(getattr(self.rc, "delta_horizon_months", 1))
        self.delta_pct = float(getattr(self.rc, "delta_pct", 0.90))
        self.label_h = int(getattr(self.rc, "label_horizon_months", 1))

    # ---- Etichetta di stress -----------------------
    @staticmethod
    def _exp_high_flag(s: pd.Series, q: float) -> pd.Series:
        # True quando s supera il proprio q-esimo percentile espandente
        thr = s.expanding(min_periods=_PCT_MIN_PERIODS).quantile(q)
        return (s > thr) & thr.notna()

    def _exp_high_delta_flag(self, s: pd.Series, q: float) -> pd.Series:
        # True quando s supera la variazione della serie (delta)
        d = s - s.shift(self.delta_h)
        thr = d.expanding(min_periods=_PCT_MIN_PERIODS).quantile(q)
        return (d > thr) & thr.notna()

    def _high_flag(self, s: pd.Series, level_pct: float, use_delta: bool) -> pd.Series:
        if use_delta:
            return self._exp_high_delta_flag(s, self.delta_pct)
        return self._exp_high_flag(s, level_pct)

    def _market_indicators(self, md: MarketData, index: pd.DatetimeIndex):
        # Volatilità realizzata (60g) e drawdown a 12 mesi del benchmark
        bench = md.benchmark.dropna().astype(float)
        bret = np.log(bench / bench.shift(1))
        rvol_d = bret.rolling(self.rc.realized_vol_window,
                              min_periods=self.rc.realized_vol_window // 2).std() * np.sqrt(_TRADING_DAYS)
        rvol_m = rvol_d.resample(_MONTH_END).last().reindex(index)
        bpx_m = bench.resample(_MONTH_END).last()
        roll_max = bpx_m.rolling(self.rc.drawdown_window_months,
                                 min_periods=max(3, self.rc.drawdown_window_months // 2)).max()
        dd_m = (bpx_m / roll_max - 1.0).reindex(index)
        return rvol_m, dd_m

    def _build_labels(self, md: MarketData, fs: FeatureSet):
        # Costruisce i 4 trigger e l'etichetta di stress (>= n trigger attivi)
        idx = fs.macro_monthly.index
        rvol, dd = self._market_indicators(md, idx)
        vix = fs.macro_monthly.get("VIX_level", pd.Series(index=idx, dtype=float))
        hy = fs.macro_monthly.get("HY_OAS_level", pd.Series(index=idx, dtype=float))
        # Modalita' per-trigger
        mode = self.trigger_mode
        vol_delta = mode == "delta"
        macro_delta = mode in ("delta", "mixed")
        trig = pd.DataFrame(index=idx)
        trig["vol"] = self._high_flag(rvol, self.rc.realized_vol_pct, vol_delta)
        trig["drawdown"] = dd < self.rc.drawdown_threshold
        trig["vix"] = self._high_flag(vix, self.rc.vix_pct, macro_delta)
        trig["hy_oas"] = self._high_flag(hy, self.rc.hy_oas_pct, macro_delta)
        trig = trig.fillna(False)
        n_active = trig.sum(axis=1)
        labels_raw = (n_active >= self.n_triggers_stress).astype(int)
        # Orizzonte dell'etichetta
        if self.label_h > 1:
            rev = labels_raw[::-1]
            labels = rev.rolling(self.label_h, min_periods=self.label_h).max()[::-1]
        else:
            labels = labels_raw.astype(float)
        return labels, trig, rvol, dd

    # ---- Feature del RF laggate ------------------------
    def _build_features(self, md: MarketData, fs: FeatureSet,
                        rvol: pd.Series, dd: pd.Series) -> pd.DataFrame:
        feat = fs.macro_monthly.copy()
        feat["MKT_RVOL"] = rvol
        feat["MKT_DD"] = dd
        return feat.shift(self.feature_lag)      # solo info fino a t-1 (no look-ahead)

    # ---- Loop rolling: RF espandente, previsione OOS ----------------------
    def run(self, md: MarketData, fs: FeatureSet) -> RegimeBook:
        labels, triggers, rvol, dd = self._build_labels(md, fs)
        features = self._build_features(md, fs, rvol, dd)
        months = list(features.index)
        min_train = self.rc.min_train_months
        prob_raw = pd.Series(index=features.index, dtype=float)
        imp_accum = pd.Series(0.0, index=features.columns)
        imp_count = 0
        for pos, t in enumerate(months):
            if pos < min_train:
                continue
            # Training espandente
            train_end = max(0, pos - self.label_h + 1)
            train_months = months[:train_end]  
            Xtr = features.loc[train_months]
            ytr = labels.loc[train_months]
            mask = Xtr.notna().all(axis=1) & ytr.notna()
            Xtr, ytr = Xtr[mask], ytr[mask]
            if len(ytr) < min_train or ytr.nunique() < 2:
                continue
            xt = features.loc[[t]]
            if xt.isna().any(axis=1).iloc[0]:
                continue
            rf = RandomForestClassifier(
                n_estimators=self.rc.n_estimators, max_depth=self.rc.max_depth,
                min_samples_leaf=self.rc.min_samples_leaf,
                class_weight=self.rc.class_weight, random_state=self.cfg.seed, n_jobs=-1)
            rf.fit(Xtr.values, ytr.values)
            stress_col = list(rf.classes_).index(1) if 1 in rf.classes_ else None
            if stress_col is None:
                continue
            prob_raw.loc[t] = float(rf.predict_proba(xt.values)[0, stress_col])
            imp_accum += pd.Series(rf.feature_importances_, index=features.columns)
            imp_count += 1
        # Lisciamento della probabilità e classificazione a soglia
        prob_smooth = prob_raw.rolling(self.rc.prob_smoothing_window, min_periods=1).mean()
        regimes = prob_smooth.apply(
            lambda p: np.nan if pd.isna(p) else ("stress" if p > self.stress_threshold else "normal"))
        importances = (imp_accum / imp_count).sort_values(ascending=False) if imp_count else pd.Series(dtype=float)
        diagnostics = self._diagnostics(labels, prob_smooth, regimes)
        logger.info("Regimi: %d mesi OOS | stress previsti=%d | accuracy OOS=%.3f | freq stress reale=%.3f",
                    int(prob_raw.notna().sum()),
                    int((regimes == "stress").sum()),
                    diagnostics.get("accuracy", float("nan")),
                    diagnostics.get("stress_freq_true", float("nan")))
        return RegimeBook(regimes=regimes.dropna(), prob_stress=prob_smooth.dropna(),
                          prob_raw=prob_raw.dropna(), labels=labels, triggers=triggers,
                          importances=importances, diagnostics=diagnostics)

    # ---- Diagnostica ------------------------------------------------------
    @staticmethod
    def _diagnostics(labels: pd.Series, prob_smooth: pd.Series,
                     regimes: pd.Series) -> Dict[str, float]:
        # Accuracy, precision, recall e frequenze di stress su periodo OOS comune
        pred = regimes.dropna().map({"normal": 0, "stress": 1})
        common = pred.index.intersection(labels.index)
        y, yhat = labels.reindex(common), pred.reindex(common)
        out: Dict[str, float] = {}
        if len(common):
            out["accuracy"] = float((y == yhat).mean())
            tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
            fn = int(((y == 1) & (yhat == 0)).sum())
            out["precision"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
            out["recall"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            out["stress_freq_true"] = float(y.mean())
            out["stress_freq_pred"] = float(yhat.mean())
            out["n_oos"] = int(len(common))
        return out

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    CONFIG.data.data_dir = Path(args.data_dir)
    CONFIG.data.fred_dir = Path(args.data_dir)
    md = DataLoader(CONFIG).load()
    fs = FeatureEngineer(CONFIG).build(md)
    rb = RegimeDetector(CONFIG).run(md, fs)
    print("\n=== DIAGNOSTICA REGIMI ===")
    for k, v in rb.diagnostics.items():
        print(f"  {k}: {round(v, 4)}")
    print("\nMesi in stress (previsti):", int((rb.regimes == 'stress').sum()),
          "su", len(rb.regimes))
    print("\n=== IMPORTANZA MEDIA FEATURE (top 10) ===")
    print(rb.importances.head(10).round(4).to_string())
    print("\n=== ULTIMI 6 MESI ===")
    tail = pd.DataFrame({"prob_stress": rb.prob_stress, "regime": rb.regimes,
                         "label_reale": rb.labels}).dropna().tail(6)
    print(tail.round(3).to_string())