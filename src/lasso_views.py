# =============================================================================
# LASSOVIEWS.PY — Blocco 3 · Generazione sistematica delle view via LASSO
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from config import Config, CONFIG
from feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

# Parametri non presenti in LassoConfig (default interni)
_WINSOR_Z = 3.0               # cap robusto degli z-score cross-section per la selezione
_OMEGA_MIN_ERRORS = 6         # errori OOS minimi per stimare Omega dagli errori
_OMEGA_FALLBACK_SCALE = 1.0   # scala del prior di incertezza (fallback)
_OMEGA_FLOOR = 1e-8           # pavimento numerico della diagonale di Omega
_LASSO_MAX_ITER = 50000
_MIN_TRAIN_ROWS = 50          # righe minime nel design per stimare il LASSO

# -----------------------------------------------------------------------------
# Strutture di output
# -----------------------------------------------------------------------------
@dataclass
class ViewSet:
    # View per una singola data di ribilanciamento t (input di Black-Litterman)
    date: pd.Timestamp     # data di ribilanciamento
    assets: List[str]      # asset con previsione valida a t (colonne di P)
    selected: List[str]    # asset con view attiva (|z| > soglia)
    P: pd.DataFrame        # matrice di picking (n_views x n_assets)
    Q: pd.Series           # vettore delle view assolute (n_views)
    omega: pd.Series       # diagonale di Omega (n_views)
    forecasts: pd.Series   # previsioni del LASSO per tutti gli asset disponibili
    zscores: pd.Series     # z-score cross-section delle previsioni
    alpha: float           # lambda scelto dalla CV
    coef: pd.Series        # coefficienti del LASSO per feature
    n_train: int           # righe usate in addestramento

@dataclass
class ViewBook:
    # Raccolta di tutte le view nel tempo, con diagnostica e percorso dei coefficienti
    views: Dict[pd.Timestamp, ViewSet] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    coef_path: pd.DataFrame = field(default_factory=pd.DataFrame)

# -----------------------------------------------------------------------------
# Generatore di view
# -----------------------------------------------------------------------------
class LassoViewGenerator:
    # Stima rolling del LASSO e costruzione di P, Q, Omega per ogni mese
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.lc = config.lasso
        # Parametri opzionali
        self.winsor_z = float(getattr(self.lc, "winsor_z", _WINSOR_Z))
        self.omega_min_errors = int(getattr(self.lc, "omega_min_errors", _OMEGA_MIN_ERRORS))
        self.omega_fallback_scale = float(getattr(self.lc, "omega_fallback_scale", _OMEGA_FALLBACK_SCALE))

    # ---- Standardizzazioni --------------------------------------------------
    @staticmethod
    def _cross_sectional_z(panel: pd.DataFrame) -> pd.DataFrame:
        # Feature di asset confrontate tra asset nello stesso mese
        g = panel.groupby(level="date")
        mu = g.transform("mean")
        sd = g.transform("std")
        z = (panel - mu) / sd
        return z.replace([np.inf, -np.inf], np.nan)

    def _macro_z_window(self, macro: pd.DataFrame, train_months: List[pd.Timestamp],
                        predict_month: pd.Timestamp) -> pd.DataFrame:
        # Standardizza le macro nel tempo con statistiche stimate solo sul training
        tr = macro.reindex(train_months)
        mu = tr.mean()
        sd = tr.std(ddof=0).replace(0.0, np.nan)
        rows = train_months + [predict_month]
        z = (macro.reindex(rows) - mu) / sd
        return z.fillna(0.0)

    # ---- Cross-validation rispettosa del tempo ------------------------------
    def _grouped_time_splits(self, row_months: np.ndarray) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        # Fold di CV per gruppi mensili
        uniq = np.array(sorted(pd.unique(row_months)))
        n_splits = min(self.lc.cv_folds, len(uniq) - 1)
        if n_splits < 2:
            return None
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr_idx, te_idx in tscv.split(uniq):
            tr_m, te_m = set(uniq[tr_idx]), set(uniq[te_idx])
            tr_rows = np.where(np.isin(row_months, list(tr_m)))[0]
            te_rows = np.where(np.isin(row_months, list(te_m)))[0]
            if len(tr_rows) and len(te_rows):
                splits.append((tr_rows, te_rows))
        return splits or None

    # ---- Costruzione del design pooled --------------------------------------
    def _build_design(self, asset_z: pd.DataFrame, macro_z: pd.DataFrame,
                      fwd: pd.DataFrame, months: List[pd.Timestamp],
                      feature_cols: List[str], macro_cols: List[str]
                      ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        # Impila le righe (asset, mese) con target realizzato in un design pooled
        rows, ys, row_months, idx = [], [], [], []
        for m in months:
            if m not in macro_z.index:
                continue
            mz = macro_z.loc[m]
            for tk in self.cfg.tickers:
                if (m, tk) not in asset_z.index:
                    continue
                az = asset_z.loc[(m, tk)]
                if az.isna().any():
                    continue
                yv = fwd.at[m, tk] if tk in fwd.columns else np.nan
                if pd.isna(yv):
                    continue
                rows.append(np.concatenate([az[feature_cols].values, mz[macro_cols].values]))
                ys.append(float(yv))
                row_months.append(m)
                idx.append((m, tk))
        if not rows:
            return pd.DataFrame(columns=feature_cols + macro_cols), np.array([]), np.array([])
        X = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["date", "ticker"]),
                         columns=feature_cols + macro_cols)
        return X, np.asarray(ys), np.asarray(row_months)

    # ---- Selezione delle view -----------------------------------------------
    def _select_views(self, forecasts: pd.Series) -> Tuple[pd.Series, List[str]]:
        # z-score cross-section robusto (winsorizzato) e selezione |z| > soglia
        f = forecasts.dropna()
        if len(f) < 2:
            return pd.Series(dtype=float), []
        mu, sd = f.mean(), f.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=f.index), []
        # Winsorizza le previsioni e ricalcola lo z robusto
        f_w = f.clip(mu - self.winsor_z * sd, mu + self.winsor_z * sd)
        mu_w, sd_w = f_w.mean(), f_w.std(ddof=0)
        z = (f_w - mu_w) / sd_w if sd_w > 0 else pd.Series(0.0, index=f.index)
        selected = z.index[z.abs() > self.lc.view_z_threshold].tolist()
        return z, selected

    # ---- Omega --------------------------------------------------------------
    def _omega_diag(self, selected: List[str], t: pd.Timestamp,
                    fcst_by_month: Dict[pd.Timestamp, pd.Series],
                    fwd: pd.DataFrame, train_months: List[pd.Timestamp]) -> pd.Series:
        # Diagonale di Omega dalla varianza degli errori OOS del tilt cross-section
        L = self.lc.omega_window_months
        past = sorted(m for m in fcst_by_month if m < t)[-L:]
        tilt_err: Dict[str, List[float]] = {tk: [] for tk in selected}
        for m in past:
            f_m = fcst_by_month[m].dropna()
            common = [tk for tk in f_m.index
                      if tk in fwd.columns and pd.notna(fwd.at[m, tk])]
            if len(common) < 2:
                continue
            f_c = f_m.reindex(common)
            r_c = fwd.loc[m, common]
            e = (f_c - f_c.mean()) - (r_c - r_c.mean())   # errore del tilt
            for tk in selected:
                if tk in e.index:
                    tilt_err[tk].append(float(e[tk]))
        omega = {}
        for tk in selected:
            errs = tilt_err[tk]
            if len(errs) >= self.omega_min_errors:
                omega[tk] = float(np.var(errs, ddof=1))
            else:
                # Fallback: varianza dei rendimenti sul training, con pavimento numerico
                rv = fwd.reindex(train_months)[tk].var(ddof=1) if tk in fwd.columns else np.nan
                omega[tk] = float(rv) * self.omega_fallback_scale if np.isfinite(rv) else _OMEGA_FLOOR
            omega[tk] = max(omega[tk], _OMEGA_FLOOR)
        return pd.Series(omega)

    # ---- Loop principale ----------------------------------------------------
    def run(self, fs: FeatureSet) -> ViewBook:
        # Genera le view per tutte le date di ribilanciamento ammissibili
        np.random.seed(self.cfg.seed)
        feature_cols = list(fs.asset_monthly.columns)
        macro_cols = list(fs.macro_monthly.columns)
        fwd = fs.forward_returns
        months = list(fwd.index)
        asset_z = self._cross_sectional_z(fs.asset_monthly)
        W, min_train = self.lc.train_window_months, self.lc.min_train_months
        views: Dict[pd.Timestamp, ViewSet] = {}
        fcst_by_month: Dict[pd.Timestamp, pd.Series] = {}
        diag_rows, coef_rows = [], {}
        for pos, t in enumerate(months):
            train_months = months[max(0, pos - W):pos]
            if len(train_months) < min_train:
                continue
            macro_z = self._macro_z_window(fs.macro_monthly, train_months, t)
            X, y, row_months = self._build_design(asset_z, macro_z, fwd, train_months,
                                                  feature_cols, macro_cols)
            if len(y) < _MIN_TRAIN_ROWS:
                continue
            splits = self._grouped_time_splits(row_months)
            model = LassoCV(n_alphas=self.lc.n_lambda, cv=splits if splits else 3,
                            max_iter=_LASSO_MAX_ITER, tol=1e-4,
                            random_state=self.cfg.seed, n_jobs=None)
            model.fit(X.values, y)
            coef = pd.Series(model.coef_, index=X.columns)
            # Previsione per il mese t (asset con feature complete)
            fc = {}
            mz_t = macro_z.loc[t]
            for tk in self.cfg.tickers:
                if (t, tk) not in asset_z.index:
                    continue
                az = asset_z.loc[(t, tk)]
                if az.isna().any():
                    continue
                xrow = np.concatenate([az[feature_cols].values, mz_t[macro_cols].values])
                fc[tk] = float(model.predict(xrow.reshape(1, -1))[0])
            forecasts = pd.Series(fc)
            if forecasts.empty:
                continue
            fcst_by_month[t] = forecasts
            # Selezione view e costruzione di Q
            z, selected = self._select_views(forecasts)
            f = forecasts.dropna()
            mu_f, sd_f = f.mean(), f.std(ddof=0)
            if np.isfinite(sd_f) and sd_f > 0:
                f_w = f.clip(mu_f - self.winsor_z * sd_f, mu_f + self.winsor_z * sd_f)
            else:
                f_w = f
            tilt = f_w - f_w.mean()
            Q = tilt.reindex(selected).copy()
            clip = self.lc.clip_view_abs
            if clip is not None:
                Q = Q.clip(-clip, clip)
            omega = self._omega_diag(selected, t, fcst_by_month, fwd, train_months)
            assets = list(forecasts.index)
            P = pd.DataFrame(0.0, index=selected, columns=assets)
            for a in selected:
                P.at[a, a] = 1.0
            views[t] = ViewSet(date=t, assets=assets, selected=selected, P=P, Q=Q,
                               omega=omega, forecasts=forecasts, zscores=z,
                               alpha=float(model.alpha_), coef=coef, n_train=len(y))
            diag_rows.append({"date": t, "n_views": len(selected), "alpha": float(model.alpha_),
                              "n_feat_selected": int((coef != 0).sum()), "n_train": len(y),
                              "n_assets": len(assets)})
            coef_rows[t] = coef
        diagnostics = (pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame())
        coef_path = (pd.DataFrame(coef_rows).T if coef_rows else pd.DataFrame())
        coef_path.index.name = "date"
        logger.info("Views generate per %d mesi (media %.1f views/mese).",
                    len(views), diagnostics["n_views"].mean() if not diagnostics.empty else 0.0)
        return ViewBook(views=views, diagnostics=diagnostics, coef_path=coef_path)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    CONFIG.data.data_dir = Path(args.data_dir)
    CONFIG.data.fred_dir = Path(args.data_dir)
    md = DataLoader(CONFIG).load()
    fs = FeatureEngineer(CONFIG).build(md)
    book = LassoViewGenerator(CONFIG).run(fs)
    print("Date con views:", len(book.views))
    print("\nDiagnostica (coda):")
    print(book.diagnostics.tail(8).round(5).to_string())
    print("\nFrequenza di selezione delle feature dal LASSO (top):")
    freq = (book.coef_path != 0).mean().sort_values(ascending=False)
    print(freq.head(12).round(3).to_string())
    last = max(book.views)
    vs = book.views[last]
    print(f"\nUltima data {last.date()}: {len(vs.selected)} views su {len(vs.assets)} asset")
    if vs.selected:
        tbl = pd.DataFrame({"Q (forecast)": vs.Q, "omega": vs.omega,
                            "z": vs.zscores.reindex(vs.selected)})
        print(tbl.round(5).to_string())