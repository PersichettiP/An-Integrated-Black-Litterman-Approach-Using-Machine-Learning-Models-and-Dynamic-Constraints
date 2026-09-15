# =============================================================================
# BLACKLITTERMAN.PY — Blocco 4 · Modello di Black-Litterman
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from config import Config, CONFIG
from data_loader import MarketData
from lasso_views import ViewBook

logger = logging.getLogger(__name__)
_MONTH_END = "ME"
_DELTA_FLOOR = 1.0        # avversione minima ammessa
_DELTA_CAP = 20.0         # cap prudenziale sulla delta di mercato
_OMEGA_FLOOR = 1e-8       # pavimento numerico su Omega
_COV_MIN_MONTHS = 36      # storia minima per includere un asset nella covarianza

# ---------------------------------------------------------------------------
# Contenitori di output
# ---------------------------------------------------------------------------
@dataclass
class BLResult:
    # Risultato Black-Litterman per un singolo mese
    date: pd.Timestamp
    assets: List[str]
    pi: pd.Series                 # prior di equilibrio
    mu_bl: pd.Series              # attese posteriori
    sigma: pd.DataFrame           # covarianza prior
    sigma_bl: pd.DataFrame        # covarianza posteriori
    delta: float                  # avversione al rischio
    n_views: int
    w_implied: pd.Series          # pesi impliciti da mu_bl

@dataclass
class BLBook:
    # Raccolta dei risultati BL nel tempo
    results: Dict[pd.Timestamp, BLResult] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------
class BlackLittermanModel:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.bl = config.bl
        self.tau = float(self.bl.tau)
        self.cov_window = int(getattr(self.bl, "cov_window_months",
                                      config.risk.vol_estimation_window))
        self.min_cov_months = int(getattr(self.bl, "min_cov_months", _COV_MIN_MONTHS))
        # Ticker risk-free per l'eccesso nella stima del delta di mercato
        cash = config.tickers_in_group("cash")
        self.rf_ticker = getattr(self.bl, "risk_free_ticker", cash[0] if cash else None)

    # ---- Rendimenti mensili ----------------------------------------------
    @staticmethod
    def _monthly_log_returns(prices_eur: pd.DataFrame) -> pd.DataFrame:
        px = prices_eur.resample(_MONTH_END).last()
        return np.log(px / px.shift(1))

    @staticmethod
    def _monthly_log_returns_series(s: pd.Series) -> pd.Series:
        px = s.resample(_MONTH_END).last()
        return np.log(px / px.shift(1))

    # ---- Covarianza Ledoit-Wolf ------------------------------------------
    def _covariance(self, window_rets: pd.DataFrame) -> Optional[pd.DataFrame]:
        # Covarianza sugli asset con storia sufficiente nella finestra
        counts = window_rets.notna().sum()
        keep = counts[counts >= self.min_cov_months].index.tolist()
        if len(keep) < 2:
            return None
        sub = window_rets[keep].dropna()
        # Se il campione comune è troppo corto, elimina gli asset con meno storia
        while len(sub) < self.min_cov_months and len(keep) > 2:
            shortest = window_rets[keep].notna().sum().idxmin()
            keep.remove(shortest)
            sub = window_rets[keep].dropna()
        if len(sub) < self.min_cov_months or len(keep) < 2:
            return None
        lw = LedoitWolf().fit(sub.values)
        return pd.DataFrame(lw.covariance_, index=keep, columns=keep)

    # ---- Delta implicita dal mercato -------------------------------------
    def _delta(self, bench_ret: pd.Series, rf_ret: pd.Series,
               window: List[pd.Timestamp]) -> float:
        if self.bl.delta_mode == "fixed":
            return float(self.bl.delta_fixed)
        b = bench_ret.reindex(window).dropna()
        if len(b) < 12:
            return float(self.bl.delta_fixed)
        rf = rf_ret.reindex(b.index).fillna(0.0) if rf_ret is not None else 0.0
        excess = b - rf
        var = float(b.var(ddof=1))
        if not np.isfinite(var) or var <= 0:
            return float(self.bl.delta_fixed)
        d = float(excess.mean()) / var
        if not np.isfinite(d) or d <= _DELTA_FLOOR:
            return float(self.bl.delta_fixed)
        return float(min(d, _DELTA_CAP))

    # ---- Pesi strategici rinormalizzati ----------------------------------
    def _w_mkt(self, assets: List[str]) -> pd.Series:
        w = pd.Series({a: self.bl.strategic_weights.get(a, 0.0) for a in assets})
        s = w.sum()
        if s <= 0:
            w = pd.Series(1.0 / len(assets), index=assets)   # fallback equal-weight
        else:
            w = w / s
        return w

    # ---- Posterior (master formula) --------------------------------------
    def _posterior(self, pi: pd.Series, sigma: pd.DataFrame,
                   P: Optional[pd.DataFrame], Q_abs: Optional[pd.Series],
                   omega: Optional[pd.Series]):
        assets = list(pi.index)
        S = sigma.loc[assets, assets].values
        tauS_inv = np.linalg.inv(self.tau * S)
        pi_v = pi.values
        if P is None or P.shape[0] == 0:
            Minv = np.linalg.inv(tauS_inv)
            mu = pi_v.copy()
        else:
            Pm = P.loc[:, assets].values
            Om_inv = np.diag(1.0 / np.maximum(omega.values, _OMEGA_FLOOR))
            A = tauS_inv + Pm.T @ Om_inv @ Pm
            Minv = np.linalg.inv(A)
            mu = Minv @ (tauS_inv @ pi_v + Pm.T @ Om_inv @ Q_abs.values)
        mu_bl = pd.Series(mu, index=assets)
        sigma_bl = pd.DataFrame(S + Minv, index=assets, columns=assets)
        return mu_bl, sigma_bl

    # ---- Loop principale --------------------------------------------------
    def run(self, md: MarketData, view_book: ViewBook) -> BLBook:
        rets_m = self._monthly_log_returns(md.prices_eur)
        bench_m = self._monthly_log_returns_series(md.benchmark)
        rf_m = rets_m[self.rf_ticker] if self.rf_ticker in rets_m.columns else None
        months = list(rets_m.index)
        W = self.cov_window
        results: Dict[pd.Timestamp, BLResult] = {}
        diag = []
        # Mesi di decisione: quelli valutati dal LASSO (allineamento con le views)
        decision_months = (list(view_book.diagnostics.index)
                           if not view_book.diagnostics.empty else months)
        for t in decision_months:
            if t not in months:
                continue
            pos = months.index(t)
            window = months[max(0, pos - W + 1):pos + 1] 
            window_rets = rets_m.loc[window]
            sigma = self._covariance(window_rets)
            if sigma is None:
                continue
            assets = list(sigma.index)
            delta = self._delta(bench_m, rf_m, window)
            w_mkt = self._w_mkt(assets)
            pi = pd.Series(delta * (sigma.values @ w_mkt.values), index=assets)
            # View del LASSO per il mese t
            vs = view_book.views.get(t)
            P = Q_abs = omega = None
            n_views = 0
            if vs is not None and vs.selected:
                sel = [a for a in vs.selected if a in assets]
                if sel:
                    P = pd.DataFrame(0.0, index=sel, columns=assets)
                    for a in sel:
                        P.at[a, a] = 1.0
                    tilt = vs.Q.reindex(sel)
                    Q_abs = pi.reindex(sel) + tilt         # ancoraggio all'equilibrio
                    omega = vs.omega.reindex(sel)
                    n_views = len(sel)
            mu_bl, sigma_bl = self._posterior(pi, sigma, P, Q_abs, omega)
            # Pesi impliciti: senza view riproducono w_mkt
            try:
                w_impl = pd.Series(
                    np.linalg.solve(delta * sigma.values, mu_bl.values), index=assets)
            except np.linalg.LinAlgError:
                w_impl = pd.Series(np.nan, index=assets)
            results[t] = BLResult(date=t, assets=assets, pi=pi, mu_bl=mu_bl,
                                  sigma=sigma, sigma_bl=sigma_bl, delta=delta,
                                  n_views=n_views, w_implied=w_impl)
            diag.append({"date": t, "delta": delta, "n_assets": len(assets),
                         "n_views": n_views, "pi_mean": float(pi.mean()),
                         "mu_bl_mean": float(mu_bl.mean()),
                         "max_abs_tilt": float((mu_bl - pi).abs().max())})
        diagnostics = pd.DataFrame(diag).set_index("date") if diag else pd.DataFrame()
        logger.info("BL: %d mesi elaborati | delta medio=%.2f | mesi con views=%d",
                    len(results), diagnostics["delta"].mean() if not diagnostics.empty else 0.0,
                    int((diagnostics["n_views"] > 0).sum()) if not diagnostics.empty else 0)
        return BLBook(results=results, diagnostics=diagnostics)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from lasso_views import LassoViewGenerator
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    CONFIG.data.data_dir = Path(args.data_dir)
    CONFIG.data.fred_dir = Path(args.data_dir)
    md = DataLoader(CONFIG).load()
    fs = FeatureEngineer(CONFIG).build(md)
    book = LassoViewGenerator(CONFIG).run(fs)
    bl = BlackLittermanModel(CONFIG).run(md, book)
    print("\n=== DIAGNOSTICA BL (coda) ===")
    print(bl.diagnostics.tail(6).round(5).to_string())
    print("\ndelta medio:", round(bl.diagnostics["delta"].mean(), 3),
          "| mesi con views:", int((bl.diagnostics["n_views"] > 0).sum()))
    # Check
    no_view = [t for t, r in bl.results.items() if r.n_views == 0]
    if no_view:
        r = bl.results[no_view[len(no_view) // 2]]
        wm = BlackLittermanModel(CONFIG)._w_mkt(r.assets)
        err = float((r.w_implied - wm).abs().max())
        print(f"\nSanity (mese senza views {r.date.date()}): max|w_implied - w_mkt| = {err:.2e}")
    # Esempio con views
    with_view = [t for t, r in bl.results.items() if r.n_views > 0]
    if with_view:
        r = bl.results[with_view[len(with_view) // 2]]
        print(f"\n=== Mese con views {r.date.date()} (delta={r.delta:.2f}, {r.n_views} views) ===")
        tab = pd.DataFrame({"pi": r.pi, "mu_bl": r.mu_bl,
                            "tilt(mu_bl-pi)": r.mu_bl - r.pi}).round(5)
        print(tab.to_string())