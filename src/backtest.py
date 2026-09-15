# =============================================================================
# BACKTEST.PY — Blocco 8 · Backtest out-of-sample e confronto tra ottimizzatori
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from config import Config, CONFIG, RISK_FREE_TICKER
from data_loader import MarketData
from black_litterman import BLBook
from regime_detection import RegimeBook
from constraints import ConstraintBuilder, ConstraintSet
from optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)
_MONTHS_PER_YEAR = 12
_MONTH_END = "ME"

# ---------------------------------------------------------------------------
# Specifica di una variante
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariantSpec:
    # Definisce un ottimizzatore alternativo: cosa cambia rispetto al framework
    name: str
    mu_source: str          # 'sample' | 'pi' | 'bl' | 'none'
    sigma_source: str       # 'lw' (Ledoit-Wolf) | 'sample'
    use_regimes: bool
    gamma_mode: str         # 'calibrated' | 'fixed' | 'equal_weight'
    design: str             # 'A' | 'B'
    unconstrained: bool = False   # True = solo long-only + pieno investimento (baseline "nuda")

NAKED_VARIANTS: List[VariantSpec] = [
    # Baseline non vincolate per la horse-race (Cap. 4): solo long-only + pieno
    # investimento. Documentano l'instabilita' dei metodi standard senza la
    # struttura di vincoli, che e' il contributo del framework.
    VariantSpec("markowitz_nudo",  "sample", "sample", False, "fixed", "B", unconstrained=True),
    VariantSpec("bl_classic_nudo", "pi",     "lw",     False, "fixed", "B", unconstrained=True),
]

DEFAULT_VARIANTS: List[VariantSpec] = [
    # Cinque strategie, usate sia per entrambe le tipologie di confronti
    # Markowitz -> + prior BL -> + views LASSO -> + regimi (full); benchmark 1/N
    VariantSpec("markowitz",     "sample", "sample", False, "fixed",        "AB"),
    VariantSpec("bl_classic",    "pi",     "lw",     False, "fixed",        "AB"),
    VariantSpec("bl_views",      "bl",     "lw",     False, "fixed",        "A"),
    VariantSpec("full",          "bl",     "lw",     True,  "fixed",        "AB"),
    VariantSpec("equal_weight", "none", "lw", False, "equal_weight", "B"),
]

# ---------------------------------------------------------------------------
# Contenitori di output
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    name: str
    weights: pd.DataFrame          # pesi decisi a fine mese t
    returns: pd.Series             # rendimenti realizzati in t+1
    gross_returns: pd.Series       # rendimenti lordi
    turnover: pd.Series
    ex_ante_vol: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class BacktestBook:
    results: Dict[str, BacktestResult] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_A: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_B: pd.DataFrame = field(default_factory=pd.DataFrame)

# ---------------------------------------------------------------------------
# Generatore dei pesi per variante
# ---------------------------------------------------------------------------
class VariantGenerator:
    # Produce i pesi mensili di una variante riusando il motore del blocco precedente
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.opt = PortfolioOptimizer(config)
        self.builder = ConstraintBuilder(config)
        self.gamma_fixed = float(config.optimizer.gamma)

    @staticmethod
    def _naked_constraints(assets: List[str]) -> ConstraintSet:
        # Vincoli minimi: long-only + pieno investimento, nessun cap ne' limite
        # di gruppo. Rappresenta le baseline "come in letteratura" (Markowitz e
        # Black-Litterman classico non vincolati), usate solo nella horse-race.
        n = len(assets)
        return ConstraintSet(assets=list(assets), regime="none",
                             lb=np.zeros(n), ub=np.ones(n), groups=[],
                             fully_invested=True, long_only=True,
                             feasible=True, reason=None)

    @staticmethod
    def _window(rets_m: pd.DataFrame, t: pd.Timestamp, window: int) -> Optional[List]:
        # Finestra rolling di stima che termina in t (incluso)
        months = list(rets_m.index)
        if t not in months:
            return None
        pos = months.index(t)
        return months[max(0, pos - window + 1):pos + 1]

    @classmethod
    def _sample_mu(cls, rets_m: pd.DataFrame, t: pd.Timestamp, assets: List[str],
                   window: int) -> Optional[np.ndarray]:
        # Media campionaria rolling sui log-rendimenti
        win = cls._window(rets_m, t, window)
        if win is None:
            return None
        mu = rets_m.loc[win, assets].mean()
        if mu.isna().any():
            return None
        return mu.to_numpy(dtype=float)

    @classmethod
    def _sample_sigma(cls, rets_m: pd.DataFrame, t: pd.Timestamp, assets: List[str],
                      window: int) -> Optional[np.ndarray]:
        # Covarianza campionaria (senza shrinkage)
        win = cls._window(rets_m, t, window)
        if win is None:
            return None
        sub = rets_m.loc[win, assets].dropna()
        if len(sub) <= len(assets):
            return None
        S = sub.cov().to_numpy(dtype=float)
        if not np.all(np.isfinite(S)):
            return None
        return (S + S.T) / 2.0

    def run(self, spec: VariantSpec, bl_book: BLBook, regime_book: RegimeBook,
            rets_m: pd.DataFrame) -> pd.DataFrame:
        rows: Dict[pd.Timestamp, pd.Series] = {}
        vols: Dict[pd.Timestamp, float] = {}
        cov_window = self.cfg.risk.vol_estimation_window
        for t in sorted(bl_book.results):
            br = bl_book.results[t]
            assets = br.assets
            # Covarianza: Ledoit-Wolf o campionaria
            if spec.sigma_source == "lw":
                Sigma = br.sigma.loc[assets, assets].to_numpy(dtype=float)
                Sigma = (Sigma + Sigma.T) / 2.0
            elif spec.sigma_source == "sample":
                Sigma = self._sample_sigma(rets_m, t, assets, cov_window)
                if Sigma is None:
                    continue
            else:
                raise ValueError(f"sigma_source sconosciuto: {spec.sigma_source}")
            # --- Equipesato ------------
            if spec.gamma_mode == "equal_weight":
                w = np.full(len(assets), 1.0 / len(assets))
                rows[t] = pd.Series(w, index=assets)
                vols[t] = self.opt._pvol(w, Sigma)
                continue
            # --- Vettore delle attese mu -------------------------------------
            if spec.mu_source == "bl":
                mu = br.mu_bl.reindex(assets).to_numpy(dtype=float)
            elif spec.mu_source == "pi":
                mu = br.pi.reindex(assets).to_numpy(dtype=float)
            elif spec.mu_source == "sample":
                mu = self._sample_mu(rets_m, t, assets, cov_window)
                if mu is None:
                    continue
            else:
                raise ValueError(f"mu_source sconosciuto: {spec.mu_source}")
            # --- Vincoli: regime del mese o regime unico ---------------------
            if spec.unconstrained:
                cs = self._naked_constraints(assets)
            else:
                regime = regime_book.regime_at(t) if spec.use_regimes else "normal"
                cs = self.builder.build(assets, regime)
            # --- Soluzione ---------------------------------------------------
            if spec.gamma_mode == "calibrated":
                w, _status, _g = self.opt.optimize_month(mu, Sigma, cs)
            else:                                    # gamma fisso
                w = self.opt._mv(mu, Sigma, cs, self.gamma_fixed)
                if w is None:
                    w = self.opt._strategic_raw(cs)
            rows[t] = pd.Series(w, index=assets)
            vols[t] = self.opt._pvol(np.asarray(w), Sigma)
        weights = pd.DataFrame(rows).T.reindex(columns=self.cfg.tickers)
        weights.index.name = "date"
        weights.attrs["ex_ante_vol"] = pd.Series(vols).sort_index()
        return weights

# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class Backtester:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.cost_bps = float(config.backtest.transaction_costs_bps)
        self.rf_ticker = RISK_FREE_TICKER

    # ---- Rendimenti semplici mensili -------------------------------------
    @staticmethod
    def _monthly_simple_returns(prices_eur: pd.DataFrame) -> pd.DataFrame:
        px = prices_eur.resample(_MONTH_END).last()
        return px.pct_change()

    # ---- Valutazione di una serie di pesi ---------------------------------
    def evaluate(self, name: str, weights: pd.DataFrame,
                 simple_rets: pd.DataFrame, rf: pd.Series) -> BacktestResult:
        w = weights.fillna(0.0)
        months = [t for t in w.index if t in simple_rets.index]
        gross, net, turn = {}, {}, {}
        prev: Optional[pd.Series] = None
        prev_date: Optional[pd.Timestamp] = None
        for t in months:
            pos = simple_rets.index.get_loc(t)
            wt = w.loc[t]
            # Turnover: confronto coi pesi precedenti derivati (drift) col mercato
            if prev is None:
                tn = 1.0
            else:
                pos_prev = simple_rets.index.get_loc(prev_date)
                if pos == pos_prev + 1:
                    r_now = simple_rets.loc[t].reindex(prev.index)
                    m = r_now.notna() & (prev.abs() > 0)
                    r_p_prev = float((prev[m] * r_now[m]).sum())
                    denom = 1.0 + r_p_prev
                    w_drift = (prev * (1.0 + r_now.fillna(0.0)) / denom
                               if abs(denom) > 1e-12 else prev)
                else:
                    w_drift = prev
                tn = float((wt - w_drift).abs().sum() / 2.0)
            cost = tn * self.cost_bps / 1e4
            turn[t] = tn
            # Rendimento realizzato: media pesata dei rendimenti semplici di t+1
            if pos + 1 < len(simple_rets.index):
                t_next = simple_rets.index[pos + 1]
                r_next = simple_rets.loc[t_next].reindex(wt.index)
                mask = r_next.notna() & (wt.abs() > 0)
                r_p = float((wt[mask] * r_next[mask]).sum())
                gross[t_next] = r_p
                net[t_next] = r_p - cost                  # costo pagato al ribilanciamento
            prev, prev_date = wt, t
        gross_s = pd.Series(gross).sort_index()
        net_s = pd.Series(net).sort_index()
        turn_s = pd.Series(turn).sort_index()
        ex_ante = weights.attrs.get("ex_ante_vol", pd.Series(dtype=float))
        metrics = self._metrics(net_s, rf, turn_s)
        return BacktestResult(name=name, weights=w, returns=net_s, gross_returns=gross_s,
                              turnover=turn_s, ex_ante_vol=ex_ante, metrics=metrics)

    # ---- Metriche ---------------------------------------------------------
    def _metrics(self, r: pd.Series, rf: pd.Series, turn: pd.Series) -> Dict[str, float]:
        # Rendimento/vol annualizzati, Sharpe in eccesso su rf, max drawdown
        if r.empty:
            return {}
        n = len(r)
        rf_a = rf.reindex(r.index).fillna(0.0)
        excess = r - rf_a
        cum = float((1.0 + r).prod())
        ann_ret = cum ** (_MONTHS_PER_YEAR / n) - 1.0
        ann_vol = float(r.std(ddof=1)) * np.sqrt(_MONTHS_PER_YEAR)
        sharpe = (float(excess.mean()) / float(excess.std(ddof=1))
                  * np.sqrt(_MONTHS_PER_YEAR)) if excess.std(ddof=1) > 0 else np.nan
        wealth = (1.0 + r).cumprod()
        max_dd = float((wealth / wealth.cummax() - 1.0).min())
        return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
                "max_dd": max_dd,
                "turnover_avg": float(turn.mean()) if not turn.empty else np.nan,
                "n_months": n}

    # ---- Loop su tutte le varianti ----------------------------------------
    def run(self, md: MarketData, bl_book: BLBook, regime_book: RegimeBook,
            variants: Optional[List[VariantSpec]] = None) -> BacktestBook:
        variants = variants or DEFAULT_VARIANTS
        simple_rets = self._monthly_simple_returns(md.prices_eur)
        log_rets = np.log(md.prices_eur.resample(_MONTH_END).last()).diff()
        rf = (simple_rets[self.rf_ticker] if self.rf_ticker in simple_rets.columns
              else pd.Series(0.0, index=simple_rets.index))
        gen = VariantGenerator(self.cfg)
        results: Dict[str, BacktestResult] = {}
        for spec in variants:
            w = gen.run(spec, bl_book, regime_book, log_rets)
            if w.empty:
                logger.warning("Variante %s: nessun peso generato.", spec.name)
                continue
            res = self.evaluate(spec.name, w, simple_rets, rf)
            res.metrics["design"] = spec.design
            results[spec.name] = res
            logger.info("%-14s | ret %.2f%% | vol %.2f%% | Sharpe %.2f | turnover %.1f%%",
                        spec.name, 100 * res.metrics["ann_return"],
                        100 * res.metrics["ann_vol"], res.metrics["sharpe"],
                        100 * res.metrics["turnover_avg"])
        summary = pd.DataFrame({k: v.metrics for k, v in results.items()}).T
        cols = ["ann_return", "ann_vol", "sharpe", "max_dd", "turnover_avg", "n_months"]
        summary = summary.reindex(columns=cols + ["design"])

        def _sub(order: List[str]) -> pd.DataFrame:
            keep = [n for n in order if n in summary.index]
            return summary.loc[keep, cols]

        summary_A = _sub(["markowitz", "bl_classic", "bl_views", "full"])
        summary_B = _sub(["equal_weight", "markowitz", "bl_classic", "full"])
        return BacktestBook(results=results, summary=summary,
                            summary_A=summary_A, summary_B=summary_B)

    # ---- Diagnostica di concentrazione (per la horse-race) ----------------
    @staticmethod
    def concentration(weights: pd.DataFrame) -> Dict[str, float]:
        # Numero effettivo di posizioni (1/HHI) e peso massimo, medi nel tempo
        w = weights.fillna(0.0).to_numpy(dtype=float)
        hhi = (w ** 2).sum(axis=1)
        eff = np.divide(1.0, hhi, out=np.full_like(hhi, np.nan), where=hhi > 0)
        wmax = np.nanmax(np.where(w > 0, w, np.nan), axis=1)
        return {"n_eff": float(np.nanmean(eff)), "w_max": float(np.nanmean(wmax))}

    # ---- Horse-race: framework vincolato vs baseline nude -----------------
    def horse_race(self, md: MarketData, bl_book: BLBook, regime_book: RegimeBook,
                   constrained: Optional[List[str]] = None) -> pd.DataFrame:
        # Tabella di confronto "come in letteratura": esegue le baseline nude e
        # le affianca alle varianti vincolate gia' calcolate, aggiungendo le
        # colonne di stabilita' (n_eff, w_max) che rendono leggibile il trade-off.
        constrained = constrained or ["full", "bl_classic", "equal_weight"]
        book = self.run(md, bl_book, regime_book)                 # varianti standard
        naked = self.run(md, bl_book, regime_book, variants=NAKED_VARIANTS)
        rows = {}
        cols = ["ann_return", "ann_vol", "sharpe", "max_dd", "turnover_avg"]
        for nm in constrained:
            if nm in book.results:
                r = book.results[nm]
                rows[nm] = {**{c: r.metrics.get(c) for c in cols},
                            **self.concentration(r.weights), "vincoli": "si"}
        for nm, r in naked.results.items():
            rows[nm] = {**{c: r.metrics.get(c) for c in cols},
                        **self.concentration(r.weights), "vincoli": "no"}
        order = constrained + list(naked.results.keys())
        df = pd.DataFrame(rows).T.reindex(index=[o for o in order if o in rows])
        return df.reindex(columns=cols + ["n_eff", "w_max", "vincoli"])

# ---------------------------------------------------------------------------
# Formattazione delle tabelle
# ---------------------------------------------------------------------------
_PCT_COLS = ["ann_return", "ann_vol", "max_dd", "turnover_avg"]

# ---------------------------------------------------------------------------
# Robustezza: sensibilità del framework alla scelta dei pesi strategici
# ---------------------------------------------------------------------------
def _internal_shares(config: Config) -> Dict[str, Dict[str, float]]:
    # Quote interne di ciascun asset entro la sua classe nei pesi strategici base
    sw = config.bl.strategic_weights
    by_cls: Dict[str, Dict[str, float]] = {}
    for a in config.universe:
        by_cls.setdefault(a.asset_class, {})[a.ticker] = sw[a.ticker]
    shares: Dict[str, Dict[str, float]] = {}
    for cls, d in by_cls.items():
        tot = sum(d.values())
        shares[cls] = {t: (w / tot if tot > 0 else 0.0) for t, w in d.items()}
    return shares

def strategic_from_macro(config: Config, macro: Dict[str, float]) -> Dict[str, float]:
    # Ricostruisce i pesi per-asset da un'allocazione macro (safe, risky, real)
    shares = _internal_shares(config)
    w: Dict[str, float] = {}
    for cls, sh in shares.items():
        for t, s in sh.items():
            w[t] = macro.get(cls, 0.0) * s
    return w

# Allocazioni macro alternative per l'analisi di robustezza del prior
DEFAULT_PRIOR_ALTERNATIVES: Dict[str, Dict[str, float]] = {
    "base 40/50/10":       {"safe": 0.40, "risky": 0.50, "real": 0.10},
    "prudente 50/40/10":   {"safe": 0.50, "risky": 0.40, "real": 0.10},
    "aggressivo 30/60/10": {"safe": 0.30, "risky": 0.60, "real": 0.10},
}

def prior_sensitivity(md: MarketData, view_book, regime_book,
                      config: Config = CONFIG,
                      alternatives: Optional[Dict[str, Dict[str, float]]] = None
                      ) -> pd.DataFrame:
    # Rilancia il framework 'full' variando l'allocazione strategica di riferimento
    import copy
    from black_litterman import BlackLittermanModel
    alternatives = alternatives or DEFAULT_PRIOR_ALTERNATIVES
    rows: Dict[str, Dict[str, float]] = {}
    for name, macro in alternatives.items():
        cfg = copy.deepcopy(config)
        cfg.bl.strategic_weights = strategic_from_macro(config, macro)
        cfg.validate()                                      # somma=1 e coerenza
        bl_p = BlackLittermanModel(cfg).run(md, view_book)
        bt_p = Backtester(cfg).run(md, bl_p, regime_book)
        m = bt_p.results["full"].metrics
        rows[name] = {k: m.get(k) for k in
                      ["ann_return", "ann_vol", "sharpe", "max_dd", "turnover_avg"]}
    return pd.DataFrame(rows).T

def format_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Converte in percentuale le colonne che lo richiedono e arrotonda
    out = df.copy()
    out[_PCT_COLS] = (out[_PCT_COLS].astype(float) * 100).round(2)
    out["sharpe"] = out["sharpe"].astype(float).round(3)
    out["n_months"] = out["n_months"].astype(int)
    return out

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
    bt = Backtester(CONFIG).run(md, bl, rb)
    print("\n=== DISEGNO A — attribuzione (forma naturale, gamma fisso; valori in %) ===")
    print(format_summary(bt.summary_A).to_string())
    print("\n=== DISEGNO B — horse race (forme naturali; valori in %) ===")
    print(format_summary(bt.summary_B).to_string())