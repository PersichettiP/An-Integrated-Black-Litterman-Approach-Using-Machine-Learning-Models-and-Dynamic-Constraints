# =============================================================================
# DIAGNOSTICS.PY — Blocco 9 · Diagnostica statistica del confronto tra strategie
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config import Config, CONFIG

logger = logging.getLogger(__name__)
_MONTHS_PER_YEAR = 12
_SEED = 42

# Finestre di crisi per l'analisi per sottoperiodi
CRISIS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "COVID 2020": ("2020-02-01", "2020-04-30"),
    "Inflazione/tassi 2022": ("2022-01-01", "2022-12-31"),
}

# ---------------------------------------------------------------------------
# Test di Ledoit-Wolf (2008) sulla differenza di Sharpe
# ---------------------------------------------------------------------------
@dataclass
class SharpeTestResult:
    name_1: str
    name_2: str
    sharpe_1: float          # annualizzato
    sharpe_2: float
    diff: float              # sharpe_1 - sharpe_2 (annualizzato)
    se: float                # errore standard HAC della differenza (annualizzato)
    p_hac: float             # p-value asintotico
    p_bootstrap: float       # p-value bootstrap studentizzato
    n_obs: int

def _hac_covariance(V: np.ndarray, lag: Optional[int] = None) -> np.ndarray:
    # Covarianza HAC (Newey-West) con kernel di Bartlett e lag automatico
    T, k = V.shape
    if lag is None:
        lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, T - 1))
    S = (V.T @ V) / T                                   # autocovarianza di ordine 0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)                   
        G = (V[j:].T @ V[:-j]) / T
        S += w * (G + G.T)
    return S

def _sharpe_stats(r1: np.ndarray, r2: np.ndarray) -> Tuple[float, float, float, float]:
    # Sharpe delle due serie, differenza e suo errore standard HAC
    T = len(r1)
    a, b = float(r1.mean()), float(r2.mean())
    c, d = float((r1 ** 2).mean()), float((r2 ** 2).mean())
    var1, var2 = c - a ** 2, d - b ** 2
    if var1 <= 0 or var2 <= 0:
        return np.nan, np.nan, np.nan, np.nan
    s1, s2 = np.sqrt(var1), np.sqrt(var2)
    sr1, sr2 = a / s1, b / s2
    diff = sr1 - sr2

    # Gradiente di f rispetto a (a, b, c, d)
    grad = np.array([
        c / var1 ** 1.5,
        -d / var2 ** 1.5,
        -a / (2.0 * var1 ** 1.5),
        b / (2.0 * var2 ** 1.5),
    ])
    V = np.column_stack([r1 - a, r2 - b, r1 ** 2 - c, r2 ** 2 - d])
    Psi = _hac_covariance(V)
    var_diff = float(grad @ Psi @ grad) / T
    se = np.sqrt(var_diff) if var_diff > 0 else np.nan
    return sr1, sr2, diff, se

def _circular_block_bootstrap_idx(T: int, block: int, rng: np.random.Generator) -> np.ndarray:
    # Indici di un ricampionamento circolare a blocchi
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])
    return idx[:T]

def sharpe_difference_test(r1: pd.Series, r2: pd.Series, name_1: str = "1",
                           name_2: str = "2", n_boot: int = 4999,
                           block: int = 5, seed: int = _SEED,
                           rf: "pd.Series | float | None" = None) -> SharpeTestResult:
    common = r1.index.intersection(r2.index)
    x = r1.reindex(common).to_numpy(dtype=float)
    y = r2.reindex(common).to_numpy(dtype=float)
    # Rendimenti in ECCESSO su rf
    if rf is not None:
        rf_v = (rf.reindex(common).fillna(0.0).to_numpy(dtype=float)
                if isinstance(rf, pd.Series) else float(rf))
        x = x - rf_v
        y = y - rf_v
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    T = len(x)
    sr1, sr2, diff, se = _sharpe_stats(x, y)
    if not np.isfinite(se) or se <= 0:
        return SharpeTestResult(name_1, name_2, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, T)
    from scipy import stats as _st
    t_obs = diff / se
    p_hac = float(2.0 * (1.0 - _st.norm.cdf(abs(t_obs))))

    # --- Bootstrap circolare a blocchi, studentizzato ---------------------
    rng = np.random.default_rng(seed)
    count = 0
    valid = 0
    for _ in range(n_boot):
        idx = _circular_block_bootstrap_idx(T, block, rng)
        xb, yb = x[idx], y[idx]
        _, _, diff_b, se_b = _sharpe_stats(xb, yb)
        if not np.isfinite(se_b) or se_b <= 0:
            continue
        valid += 1
        # Statistica centrata sulla stima campionaria (H0 imposta nel bootstrap)
        if abs((diff_b - diff) / se_b) >= abs(t_obs):
            count += 1
    p_boot = (count + 1) / (valid + 1) if valid > 0 else np.nan
    k = np.sqrt(_MONTHS_PER_YEAR)                    
    return SharpeTestResult(name_1=name_1, name_2=name_2,
                            sharpe_1=sr1 * k, sharpe_2=sr2 * k,
                            diff=diff * k, se=se * k,
                            p_hac=p_hac, p_bootstrap=p_boot, n_obs=T)

# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> Tuple[float, float]:
    # Probabilità che lo Sharpe vero superi il benchmark, corretta per skew/kurtosi
    from scipy import stats as _st
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return np.nan, np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    if sd <= 0:
        return np.nan, np.nan
    sr = mu / sd                                       # Sharpe mensile
    g = _st.skew(r, bias=False)
    kurt = _st.kurtosis(r, fisher=False, bias=False)   # kurtosis NON in eccesso
    var_sr = (1.0 - g*sr + (kurt - 1.0)/4.0 * sr**2) / (n - 1.0)
    if not np.isfinite(var_sr) or var_sr <= 0:
        return sr*np.sqrt(_MONTHS_PER_YEAR), np.nan
    sr_b = sr_benchmark / np.sqrt(_MONTHS_PER_YEAR)
    psr = float(_st.norm.cdf((sr - sr_b) / np.sqrt(var_sr)))
    return sr*np.sqrt(_MONTHS_PER_YEAR), psr

# ---------------------------------------------------------------------------
# Rischio di coda: VaR, Expected Shortfall, Modified VaR
# ---------------------------------------------------------------------------
def var_historical(returns, alpha: float = 0.99) -> float:
    # VaR storico (quantile empirico della coda sinistra)
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    if len(r) < 5:
        return np.nan
    return float(-np.quantile(r, 1.0 - alpha))

def expected_shortfall(returns, alpha: float = 0.975) -> float:
    # Perdita attesa condizionata a superare il VaR (media della coda)
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    if len(r) < 5:
        return np.nan
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    return float(-tail.mean()) if len(tail) else np.nan

def var_modified(returns, alpha: float = 0.99) -> float:
    # VaR di Cornish-Fisher: corregge il VaR normale per skew e kurtosi
    from scipy import stats as _st
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    if len(r) < 5:
        return np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    S = _st.skew(r, bias=False)
    K = _st.kurtosis(r, fisher=True, bias=False)        # kurtosi in eccesso
    z = _st.norm.ppf(1.0 - alpha)
    zcf = (z + (z**2 - 1)*S/6.0 + (z**3 - 3*z)*K/24.0 - (2*z**3 - 5*z)*S**2/36.0)
    return float(-(mu + sd*zcf))

# ---------------------------------------------------------------------------
# Distribuzione del drawdown via block bootstrap
# ---------------------------------------------------------------------------
def _max_dd_and_tuw(r: np.ndarray) -> Tuple[float, int]:
    # Max drawdown e Time-under-Water (mesi consecutivi sotto il picco)
    w = np.cumprod(1.0 + r)
    dd = w / np.maximum.accumulate(w) - 1.0
    mdd = float(dd.min())
    under = dd < 0
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return mdd, longest

def drawdown_distribution(returns, n_boot: int = 2000, block: int = 5,
                          seed: int = _SEED) -> Dict[str, float]:
    # Distribuzione bootstrap di MDD e TuW e percentile del valore osservato
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    T = len(r)
    if T < 12:
        return {}
    mdd_obs, tuw_obs = _max_dd_and_tuw(r)
    rng = np.random.default_rng(seed)
    mdds = np.empty(n_boot); tuws = np.empty(n_boot)
    for b in range(n_boot):
        rb = r[_circular_block_bootstrap_idx(T, block, rng)]
        mdds[b], tuws[b] = _max_dd_and_tuw(rb)
    p = lambda a, q: float(np.percentile(a, q))
    return {
        "mdd_obs": mdd_obs * 100,
        "mdd_median": p(mdds, 50) * 100,
        "mdd_p05": p(mdds, 5) * 100,       
        "mdd_p95": p(mdds, 95) * 100,       
        "tuw_obs": float(tuw_obs),
        "tuw_median": p(tuws, 50),
        "tuw_p95": p(tuws, 95),
        "obs_percentile": float((mdds <= mdd_obs).mean()) * 100,
    }

# ---------------------------------------------------------------------------
# Contenitore e orchestrazione della diagnostica
# ---------------------------------------------------------------------------
@dataclass
class DiagnosticsBook:
    sharpe_tests: pd.DataFrame = field(default_factory=pd.DataFrame)
    subperiods: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    crisis: pd.DataFrame = field(default_factory=pd.DataFrame)
    psr: pd.DataFrame = field(default_factory=pd.DataFrame)
    tail_risk: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_dist: pd.DataFrame = field(default_factory=pd.DataFrame)
    stress_tail: pd.DataFrame = field(default_factory=pd.DataFrame)
    recovery: pd.DataFrame = field(default_factory=pd.DataFrame)

class PerformanceDiagnostics:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config

    # ---- Metriche su un sottoinsieme di mesi ------------------------------
    @staticmethod
    def _stats(r: pd.Series, rf: pd.Series) -> Dict[str, float]:
        # Rendimento/vol annualizzati, Sharpe in eccesso e max drawdown su r
        if len(r) < 2:
            return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                    "max_dd": np.nan, "n_months": len(r)}
        ex = r - rf.reindex(r.index).fillna(0.0)
        n = len(r)
        cum = float((1.0 + r).prod())
        wealth = (1.0 + r).cumprod()
        sd = float(ex.std(ddof=1))
        return {
            "ann_return": cum ** (_MONTHS_PER_YEAR / n) - 1.0,
            "ann_vol": float(r.std(ddof=1)) * np.sqrt(_MONTHS_PER_YEAR),
            "sharpe": (float(ex.mean()) / sd * np.sqrt(_MONTHS_PER_YEAR)) if sd > 0 else np.nan,
            "max_dd": float((wealth / wealth.cummax() - 1.0).min()),
            "n_months": n,
        }

    # ---- 1b. Probabilistic Sharpe Ratio (affidabilita' del singolo Sharpe) ---
    def psr_table(self, book, rf: pd.Series, sr_benchmarks=(0.0, 0.5)) -> pd.DataFrame:
        from scipy import stats as _st
        rows = []
        for name, res in book.results.items():
            r = res.returns
            ex = r - rf.reindex(r.index).fillna(0.0)
            sr_ann, _ = probabilistic_sharpe_ratio(ex, 0.0)
            row = {"strategia": name, "sharpe": sr_ann,
                   "skew": float(_st.skew(ex.dropna(), bias=False)),
                   "kurtosi": float(_st.kurtosis(ex.dropna(), fisher=True, bias=False)),
                   "n_mesi": int(ex.notna().sum())}
            for b in sr_benchmarks:
                _, psr = probabilistic_sharpe_ratio(ex, b)
                row[f"PSR(>{b:g})"] = psr
            rows.append(row)
        return pd.DataFrame(rows).set_index("strategia") if rows else pd.DataFrame()

    # ---- 1c. Rischio di coda: VaR 99%, ES 97.5%, Modified VaR ------
    def tail_risk_table(self, book) -> pd.DataFrame:
        rows = []
        for name, res in book.results.items():
            r = res.returns.dropna()
            vh = var_historical(r, 0.99)
            es = expected_shortfall(r, 0.975)
            vm = var_modified(r, 0.99)
            rows.append({"strategia": name,
                         "VaR99_hist": vh * 100 if np.isfinite(vh) else np.nan,
                         "ES97.5": es * 100 if np.isfinite(es) else np.nan,
                         "VaR99_mod": vm * 100 if np.isfinite(vm) else np.nan,
                         "coda(mod-hist)": (vm - vh) * 100 if np.isfinite(vm) and np.isfinite(vh) else np.nan})
        return pd.DataFrame(rows).set_index("strategia") if rows else pd.DataFrame()

    # ---- 1d. Distribuzione bootstrap del drawdown ---------------------------
    def drawdown_dist_table(self, book, n_boot: int = 2000) -> pd.DataFrame:
        rows = []
        for name, res in book.results.items():
            d = drawdown_distribution(res.returns.dropna(), n_boot=n_boot)
            if not d:
                continue
            rows.append({"strategia": name,
                         "MDD_oss": d["mdd_obs"], "MDD_mediana": d["mdd_median"],
                         "MDD_p05(peggio)": d["mdd_p05"], "MDD_p95(meglio)": d["mdd_p95"],
                         "TuW_oss(mesi)": d["tuw_obs"], "TuW_p95": d["tuw_p95"],
                         "pctile_oss": d["obs_percentile"]})
        return pd.DataFrame(rows).set_index("strategia") if rows else pd.DataFrame()

    # ---- 1. Test di significativita' --------------------------------------
    def sharpe_tests(self, book, rf: pd.Series, pairs: List[Tuple[str, str]],
                     n_boot: int = 4999) -> pd.DataFrame:
        # Test sulla differenza di Sharpe per ogni coppia (su rendimenti in eccesso)
        rows = []
        for a, b in pairs:
            if a not in book.results or b not in book.results:
                continue
            ra = book.results[a].returns
            rb = book.results[b].returns
            ex_a = ra - rf.reindex(ra.index).fillna(0.0)
            ex_b = rb - rf.reindex(rb.index).fillna(0.0)
            res = sharpe_difference_test(ex_a, ex_b, a, b, n_boot=n_boot)
            rows.append({
                "strategia_1": res.name_1, "strategia_2": res.name_2,
                "sharpe_1": res.sharpe_1, "sharpe_2": res.sharpe_2,
                "differenza": res.diff, "se": res.se,
                "p_hac": res.p_hac, "p_bootstrap": res.p_bootstrap,
                "n_obs": res.n_obs,
            })
        return pd.DataFrame(rows)

    # ---- 2. Sottoperiodi ---------------------------------------------------
    def subperiods(self, book, rf: pd.Series) -> pd.DataFrame:
        # Rendimento annualizzato per anno solare e strategia
        rows = {}
        for name, res in book.results.items():
            r = res.returns
            for year, grp in r.groupby(r.index.year):
                rows.setdefault(name, {})[str(year)] = self._stats(grp, rf)["ann_return"]
        df = pd.DataFrame(rows).T
        return df.reindex(columns=sorted(df.columns))

    def crisis_windows(self, book, rf: pd.Series,
                       windows: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
        # Rendimento cumulato e max drawdown in ciascuna finestra di crisi
        windows = windows or CRISIS_WINDOWS
        rows = {}
        for name, res in book.results.items():
            r = res.returns
            for label, (lo, hi) in windows.items():
                sub = r.loc[(r.index >= lo) & (r.index <= hi)]
                if sub.empty:
                    continue
                cum = float((1.0 + sub).prod() - 1.0) 
                dd = float(((1 + sub).cumprod() / (1 + sub).cumprod().cummax() - 1).min())
                rows.setdefault(name, {})[f"{label} rend."] = cum
                rows.setdefault(name, {})[f"{label} maxDD"] = dd
        return pd.DataFrame(rows).T

    # ---- 3. Condizionamento ai regimi -------------------------------------
    def by_regime(self, book, rf: pd.Series, regime_book) -> pd.DataFrame:
        # Metriche separate per regime, allineando il regime deciso a t col rendimento t+1
        regimes = regime_book.regimes
        rows = []
        for name, res in book.results.items():
            r = res.returns                              
            # regime deciso a t -> shift in avanti di un mese
            reg_shift = regimes.copy()
            reg_shift.index = pd.Index([d for d in regimes.index])
            mapping = {}
            months = list(r.index)
            for t_dec, reg in regimes.items():
                nxt = [m for m in months if m > t_dec]
                if nxt:
                    mapping[nxt[0]] = reg
            reg_aligned = pd.Series(mapping).reindex(r.index)
            for reg in ("normal", "stress"):
                sub = r[reg_aligned == reg]
                if len(sub) < 2:
                    continue
                st = self._stats(sub, rf)
                rows.append({"strategia": name, "regime": reg, **st})
        return pd.DataFrame(rows).set_index(["strategia", "regime"]) if rows else pd.DataFrame()

    # ---- 4. Analisi di coda condizionata al regime di stress --------------
    def stress_tail(self, book, rf: pd.Series, regime_book,
        alpha_var: float = 0.05, alpha_es: float = 0.025) -> pd.DataFrame:
        # Metriche di coda calcolate SOLO sui mesi di stress (regime deciso a t
        # allineato col rendimento t+1). Serve a documentare il beneficio del
        # de-risking la' dove dovrebbe manifestarsi: perdita massima, coda
        # sinistra (VaR/CVaR storici) e mese peggiore.
        # VaR storico al 95% ed Expected Shortfall (CVaR) al 97,5%, coerenti
        # con le soglie adottate in tail_risk_table (§4.2).
        regimes = regime_book.regimes
        rows = []
        for name, res in book.results.items():
            r = res.returns
            # allineamento regime(t) -> rendimento(t+1), come in by_regime
            months = list(r.index)
            mapping = {}
            for t_dec, reg in regimes.items():
                nxt = [m for m in months if m > t_dec]
                if nxt:
                    mapping[nxt[0]] = reg
            reg_aligned = pd.Series(mapping).reindex(r.index)
            sub = r[reg_aligned == "stress"].dropna()
            if len(sub) < 3:
                continue
            wealth = (1.0 + sub).cumprod()
            var = float(np.quantile(sub.values, alpha_var))       # VaR storico al 95%
            es_thresh = float(np.quantile(sub.values, alpha_es))  # soglia al 97,5% per l'ES
            tail_es = sub[sub <= es_thresh]
            cvar = float(tail_es.mean()) if len(tail_es) else es_thresh  # Expected Shortfall al 97,5%
            ex = sub - rf.reindex(sub.index).fillna(0.0)
            sd = float(ex.std(ddof=1))
            rows.append({
                "strategia": name,
                "n_stress": int(len(sub)),
                "rend_medio": float(sub.mean()),
                "sharpe_stress": (float(ex.mean()) / sd * np.sqrt(_MONTHS_PER_YEAR)) if sd > 0 else np.nan,
                "max_dd_stress": float((wealth / wealth.cummax() - 1.0).min()),
                f"VaR_{int((1-alpha_var)*100)}": var,
                f"CVaR_{int((1-alpha_es)*100)}": cvar,
                "mese_peggiore": float(sub.min()),
                "hit_rate": float((sub > 0).mean()),
            })
        return pd.DataFrame(rows).set_index("strategia") if rows else pd.DataFrame()

    # ---- 5. Tempo di recupero dopo il drawdown massimo --------------------
    def recovery_analysis(self, book) -> pd.DataFrame:
        # Per ciascuna strategia: profondita' del max drawdown, durata della
        # discesa (picco->valle) e tempo di recupero (valle->nuovo massimo).
        # Il de-risking dovrebbe accorciare discesa e recupero: qui si verifica.
        rows = []
        for name, res in book.results.items():
            r = res.returns.dropna()
            if len(r) < 3:
                continue
            wealth = (1.0 + r).cumprod()
            peak = wealth.cummax()
            dd = wealth / peak - 1.0
            trough = dd.idxmin()                       # data del minimo
            depth = float(dd.min())
            # picco che precede il valle
            pk_val = float(peak.loc[trough])
            pre = wealth.loc[:trough]
            peak_date = pre[pre >= pk_val - 1e-12].index[0]
            # recupero: primo mese dopo il valle in cui si torna al picco
            post = wealth.loc[trough:]
            rec = post[post >= pk_val - 1e-12]
            rec_date = rec.index[0] if len(rec) else None
            months_to_trough = sum(1 for m in wealth.index if peak_date < m <= trough)
            months_to_recover = (sum(1 for m in wealth.index if trough < m <= rec_date)
                                 if rec_date is not None else np.nan)
            rows.append({
                "strategia": name,
                "max_dd": depth,
                "data_picco": peak_date.strftime("%Y-%m"),
                "data_valle": trough.strftime("%Y-%m"),
                "mesi_discesa": int(months_to_trough),
                "mesi_recupero": (int(months_to_recover)
                                  if months_to_recover == months_to_recover else np.nan),
                "recuperato": rec_date is not None,
            })
        return pd.DataFrame(rows).set_index("strategia") if rows else pd.DataFrame()

    # ---- Esecuzione completa ----------------------------------------------
    def run(self, book, md, regime_book, pairs: Optional[List[Tuple[str, str]]] = None,
            n_boot: int = 4999) -> DiagnosticsBook:
        # Costruisce rf da XEON ed esegue tutte le diagnostiche in un unico book
        from config import RISK_FREE_TICKER
        px = md.prices_eur.resample("ME").last()
        simple = px.pct_change()
        rf = (simple[RISK_FREE_TICKER] if RISK_FREE_TICKER in simple.columns
              else pd.Series(0.0, index=simple.index))
        pairs = pairs or [
            # Confronto A — ogni gradino della scala di attribuzione
            ("bl_classic", "markowitz"),
            ("bl_views", "bl_classic"),
            ("full", "bl_views"),
            # Confronto B — framework vs benchmark
            ("full", "equal_weight"),
            ("full", "markowitz"),
            ("full", "bl_classic"),
        ]
        return DiagnosticsBook(
            sharpe_tests=self.sharpe_tests(book, rf, pairs, n_boot=n_boot),
            subperiods=self.subperiods(book, rf),
            crisis=self.crisis_windows(book, rf),
            by_regime=self.by_regime(book, rf, regime_book),
            psr=self.psr_table(book, rf),
            tail_risk=self.tail_risk_table(book),
            drawdown_dist=self.drawdown_dist_table(book, n_boot=n_boot),
            stress_tail=self.stress_tail(book, rf, regime_book),
            recovery=self.recovery_analysis(book),
        )

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from lasso_views import LassoViewGenerator
    from black_litterman import BlackLittermanModel
    from regime_detection import RegimeDetector
    from backtest import Backtester
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    CONFIG.data.data_dir = Path(args.data_dir)
    CONFIG.data.fred_dir = Path(args.data_dir)
    md = DataLoader(CONFIG).load()
    fs = FeatureEngineer(CONFIG).build(md)
    vb = LassoViewGenerator(CONFIG).run(fs)
    bl = BlackLittermanModel(CONFIG).run(md, vb)
    rb = RegimeDetector(CONFIG).run(md, fs)
    bt = Backtester(CONFIG).run(md, bl, rb)
    dg = PerformanceDiagnostics(CONFIG).run(bt, md, rb)
    print("\n=== TEST DI LEDOIT-WOLF SULLE DIFFERENZE DI SHARPE ===")
    print(dg.sharpe_tests.round(4).to_string(index=False))
    print("\n=== PROBABILISTIC SHARPE RATIO (affidabilita' del singolo Sharpe) ===")
    print(dg.psr.round(3).to_string())
    print("\n=== RISCHIO DI CODA (mensile, %; VaR 99% / ES 97.5% - livelli FRTB) ===")
    print(dg.tail_risk.round(2).to_string())
    print("\n=== DISTRIBUZIONE DEL DRAWDOWN (block bootstrap, %) ===")
    print(dg.drawdown_dist.round(2).to_string())
    print("\n=== RENDIMENTI PER ANNO SOLARE (%) ===")
    print((dg.subperiods * 100).round(2).to_string())
    print("\n=== FINESTRE DI CRISI (%) ===")
    print((dg.crisis * 100).round(2).to_string())
    print("\n=== PERFORMANCE CONDIZIONATA AL REGIME ===")
    print(dg.by_regime.round(4).to_string())