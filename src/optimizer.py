# =============================================================================
# OPTIMIZER.PY — Blocco 7 · Ottimizzatore di portafoglio media-varianza
# =============================================================================
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cvxpy as cp
from config import Config, CONFIG
from black_litterman import BLBook
from regime_detection import RegimeBook
from constraints import ConstraintBuilder, ConstraintSet

logger = logging.getLogger(__name__)
_MONTHS_PER_YEAR = 12
# Bracket e tolleranze per la calibrazione di gamma
_GAMMA_MIN = 1e-3
_GAMMA_MAX = 1e3
_MAX_ITER = 40
_VOL_TOL = 1e-4         # tolleranza sulla vol annualizzata

# ---------------------------------------------------------------------------
# Contenitori di output
# ---------------------------------------------------------------------------
@dataclass
class OptimizeResult:
    date: pd.Timestamp
    assets: List[str]
    weights: pd.Series
    regime: str
    status: str                 # optimal | vol_below_target | vol_above_target | strategic_projected | strategic_raw
    ex_ante_vol: float          # volatilità annualizzata ex-ante del portafoglio scelto
    gamma: float                # coefficiente di avversione calibrato
    target_hit: bool
    n_views: int

@dataclass
class OptimizerBook:
    results: Dict[pd.Timestamp, OptimizeResult] = field(default_factory=dict)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

# ---------------------------------------------------------------------------
# Ottimizzatore
# ---------------------------------------------------------------------------
class PortfolioOptimizer:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.builder = ConstraintBuilder(config)
        self.target_vol = float(config.risk.target_vol_annual)
        self._solver = self._pick_solver(config.optimizer.solver)

    @staticmethod
    def _pick_solver(preferred: str) -> str:
        # Primo solver disponibile tra il preferito e i fallback
        available = cp.installed_solvers()
        for s in (preferred, "ECOS", "CLARABEL", "SCS"):
            if s in available:
                return s
        return available[0]

    # ---- Vincoli base ----------------------------------------------------
    def _base_constraints(self, w: cp.Variable, cs: ConstraintSet) -> list:
        # Traduce il ConstraintSet in vincoli cvxpy
        cons = [w <= cs.ub]
        if cs.long_only:
            cons.append(w >= 0)
        if cs.fully_invested:
            cons.append(cp.sum(w) == 1)
        for g in cs.groups:
            expr = cp.sum(w[g.members])
            if g.lower is not None:
                cons.append(expr >= g.lower)
            if g.upper is not None:
                cons.append(expr <= g.upper)
        return cons

    def _solve(self, objective, constraints) -> Optional[np.ndarray]:
        # Risolve il problema e ripulisce numericamente i pesi
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=self._solver)
        except Exception:
            try:
                prob.solve(solver="SCS")
            except Exception:
                return None
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        w = prob.variables()[0].value
        if w is None:
            return None
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        s = w.sum()
        return w / s if s > 1e-9 else None

    # ---- Media-varianza a gamma dato --------------------------------------
    def _mv(self, mu, Sigma, cs, gamma: float) -> Optional[np.ndarray]:
        w = cp.Variable(len(cs.assets))
        S = cp.psd_wrap(Sigma)
        cons = self._base_constraints(w, cs)
        obj = cp.Maximize(mu @ w - (gamma / 2.0) * cp.quad_form(w, S))
        return self._solve(obj, cons)

    def _pvol(self, w: np.ndarray, Sigma: np.ndarray) -> float:
        # Volatilità annualizzata ex-ante
        return float(np.sqrt(max(float(w @ Sigma @ w), 0.0)) * np.sqrt(_MONTHS_PER_YEAR))

    # ---- Fallback strategici ---------------------------------------------
    def _project_strategic(self, cs) -> Optional[np.ndarray]:
        # Proietta i pesi strategici sull'insieme ammissibile
        w_strat = self._strategic_vector(cs.assets)
        w = cp.Variable(len(cs.assets))
        cons = self._base_constraints(w, cs)
        obj = cp.Minimize(cp.sum_squares(w - w_strat))
        return self._solve(obj, cons)

    def _strategic_vector(self, assets: List[str]) -> np.ndarray:
        sw = self.cfg.bl.strategic_weights
        w = np.array([sw.get(a, 0.0) for a in assets], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.full(len(assets), 1.0 / len(assets))

    def _strategic_raw(self, cs) -> np.ndarray:
        # Pesi strategici troncati ai cap
        w = np.clip(self._strategic_vector(cs.assets), 0.0, cs.ub)
        s = w.sum()
        return w / s if s > 1e-9 else np.full(len(cs.assets), 1.0 / len(cs.assets))

    # ---- Ottimizzazione di un mese con calibrazione di gamma --------------
    def optimize_month(self, mu: np.ndarray, Sigma: np.ndarray,
                       cs: ConstraintSet) -> Tuple[np.ndarray, str, float]:
        if not cs.feasible:
            return self._strategic_raw(cs), "strategic_raw", float("nan")
        target = self.target_vol
        # Estremi del bracket: gamma_min -> vol massima; gamma_max -> vol minima
        w_lo = self._mv(mu, Sigma, cs, _GAMMA_MIN)
        w_hi = self._mv(mu, Sigma, cs, _GAMMA_MAX)
        if w_lo is None or w_hi is None:
            w = self._project_strategic(cs)
            if w is not None:
                return w, "strategic_projected", float("nan")
            return self._strategic_raw(cs), "strategic_raw", float("nan")
        vol_lo = self._pvol(w_lo, Sigma)        
        vol_hi = self._pvol(w_hi, Sigma)         
        # Mese calmo: neanche col massimo rischio si arriva al target
        if vol_lo <= target + _VOL_TOL:
            return w_lo, "vol_below_target", _GAMMA_MIN
        # Crisi: neanche il min-variance scende al target
        if vol_hi >= target - _VOL_TOL:
            return w_hi, "vol_above_target", _GAMMA_MAX
        # Bisezione su gamma per centrare il target
        lo, hi = _GAMMA_MIN, _GAMMA_MAX
        w_star, g_star, best = w_hi, _GAMMA_MAX, abs(vol_hi - target)
        for _ in range(_MAX_ITER):
            g = math.sqrt(lo * hi)          # media geometrica
            w = self._mv(mu, Sigma, cs, g)
            if w is None:
                break
            v = self._pvol(w, Sigma)
            if abs(v - target) < best:
                w_star, g_star, best = w, g, abs(v - target)
            if abs(v - target) <= _VOL_TOL:
                break
            if v > target:          # troppo rischioso -> piu' avversione
                lo = g
            else:                   # troppo prudente -> meno avversione
                hi = g
        return w_star, "optimal", g_star

    # ---- Loop principale --------------------------------------------------
    def run(self, bl_book: BLBook, regime_book: RegimeBook) -> OptimizerBook:
        results: Dict[pd.Timestamp, OptimizeResult] = {}
        wrows: Dict[pd.Timestamp, pd.Series] = {}
        diag = []
        for t in sorted(bl_book.results):
            br = bl_book.results[t]
            assets = br.assets
            mu = br.mu_bl.reindex(assets).to_numpy(dtype=float)
            Sigma = br.sigma.loc[assets, assets].to_numpy(dtype=float)
            Sigma = (Sigma + Sigma.T) / 2.0
            regime = regime_book.regime_at(t) if regime_book is not None else "normal"
            cs = self.builder.build(assets, regime)
            w, status, gamma = self.optimize_month(mu, Sigma, cs)
            wser = pd.Series(w, index=assets)
            ex_ante_vol = self._pvol(w, Sigma)
            target_hit = abs(ex_ante_vol - self.target_vol) <= 1e-3
            results[t] = OptimizeResult(date=t, assets=assets, weights=wser, regime=regime,
                                        status=status, ex_ante_vol=ex_ante_vol, gamma=gamma,
                                        target_hit=target_hit, n_views=br.n_views)
            wrows[t] = wser
            diag.append({"date": t, "regime": regime, "status": status,
                         "ex_ante_vol": ex_ante_vol, "gamma": gamma,
                         "target_hit": target_hit, "n_views": br.n_views,
                         "n_assets": len(assets)})
        weights = pd.DataFrame(wrows).T.reindex(columns=self.cfg.tickers)
        weights.index.name = "date"
        diagnostics = pd.DataFrame(diag).set_index("date") if diag else pd.DataFrame()
        if not diagnostics.empty:
            logger.info("Optimizer: %d mesi | optimal=%d | vol media=%.3f | target centrato in %d mesi",
                        len(results), int((diagnostics["status"] == "optimal").sum()),
                        diagnostics["ex_ante_vol"].mean(), int(diagnostics["target_hit"].sum()))
        return OptimizerBook(results=results, weights=weights, diagnostics=diagnostics)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from lasso_views import LassoViewGenerator
    from black_litterman import BlackLittermanModel
    from regime_detection import RegimeDetector
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
    rb = RegimeDetector(CONFIG).run(md, fs)
    opt = PortfolioOptimizer(CONFIG).run(bl, rb)
    d = opt.diagnostics
    print("\n=== DIAGNOSTICA OTTIMIZZATORE (coda) ===")
    print(d.tail(8).round(4).to_string())
    print("\nStati:", d["status"].value_counts().to_dict())
    print("Vol ex-ante media:", round(d["ex_ante_vol"].mean(), 4),
          "| target centrato:", int(d["target_hit"].sum()), "su", len(d))
    print("Vol media per regime:")
    print(d.groupby("regime")["ex_ante_vol"].mean().round(4).to_string())
    last = max(opt.results); r = opt.results[last]
    print(f"\nPesi a {last.date()} (regime {r.regime}, vol {r.ex_ante_vol:.3f}, gamma {r.gamma:.3g}):")
    print(r.weights[r.weights > 1e-4].round(4).to_string())