# =============================================================================
# PLOTS.PY — Libreria di visualizzazione del framework
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

from config import CONFIG, RISK_FREE_TICKER

PALETTE = {
    "markowitz":    "#4C72B0",
    "bl_classic":   "#DD8452",
    "bl_views":     "#55A868",
    "equal_weight": "#8C8C8C",
    "full":         "#8172B3",
    "stress":       "#C44E52",
    "accent":       "#08306B",
    "grid":         "#D9D9D9",
    "ink":          "#1A1A1A",
}

# etichette leggibili e ordine "a scala" delle strategie
LABELS = {
    "markowitz": "Markowitz", "bl_classic": "Black-Litterman",
    "bl_views": "BL + views LASSO", "equal_weight": "Equipesato (1/N)",
    "full": "Framework completo",
}
LADDER = ["markowitz", "bl_classic", "bl_views", "full"]
COMPARE = ["markowitz", "bl_classic", "bl_views", "equal_weight", "full"]

# colori per classe di attivo (torta e barre della scheda di allocazione)
CLASS_COLORS = {"safe": "#4C72B0", "risky": "#C44E52", "real": "#C9A227"}
CLASS_LABELS = {"safe": "Difensivi", "risky": "Rischiosi", "real": "Reali (oro)"}

_MPY = 12
_DPY = 252
_INITIAL_CAPITAL = 1_000_000.0

def apply_theme() -> None:
    """Imposta uno stile matplotlib sobrio e professionale (idempotente)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# ---------------------------------------------------------------------------
# Utilità statistiche condivise
# ---------------------------------------------------------------------------
def _hac(V: np.ndarray, lag: Optional[int] = None) -> np.ndarray:
    T = len(V)
    if lag is None:
        lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lag = max(0, min(lag, T - 1))
    S = (V.T @ V) / T
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)
        G = (V[j:].T @ V[:-j]) / T
        S += w * (G + G.T)
    return S

def _sharpe_se(excess: np.ndarray) -> Tuple[float, float]:
    """Sharpe annualizzato e suo errore standard HAC (Lo 2002, delta method)."""
    x = np.asarray(excess, float)
    x = x[np.isfinite(x)]
    T = len(x)
    a, c = x.mean(), (x ** 2).mean()
    var = c - a ** 2
    if var <= 0:
        return np.nan, np.nan
    grad = np.array([c / var ** 1.5, -a / (2 * var ** 1.5)])
    V = np.column_stack([x - a, x ** 2 - c])
    se = np.sqrt(grad @ _hac(V) @ grad / T)
    return a / np.sqrt(var) * np.sqrt(_MPY), se * np.sqrt(_MPY)

# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------
class Charts:
    def __init__(self, backtest, md=None, rb=None, config=CONFIG,
                 outdir: str = ".") -> None:
        self.bt = backtest
        self.md = md
        self.rb = rb
        self.cfg = config
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._suppress = False        # True durante save_all: salva senza mostrare
        apply_theme()

    # ---- helper interni ---------------------------------------------------
    def _wealth(self, key: str, base: float = 100.0) -> pd.Series:
        r = self.bt.results[key].returns.sort_index()
        w = base * (1.0 + r).cumprod()
        start = w.index[0] - pd.offsets.MonthEnd(1)
        return pd.concat([pd.Series({start: base}), w])

    def _drawdown(self, key: str) -> pd.Series:
        r = self.bt.results[key].returns.sort_index()
        w = (1.0 + r).cumprod()
        return (w / w.cummax() - 1.0) * 100.0

    def _rf(self) -> pd.Series:
        s = self.md.prices_eur.resample("ME").last().pct_change()
        return s[RISK_FREE_TICKER] if RISK_FREE_TICKER in s.columns else pd.Series(0.0, index=s.index)

    def _save(self, fig, name: str) -> None:
        fig.savefig(self.outdir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.outdir / f"{name}.png", dpi=200, bbox_inches="tight")

    def _keys(self) -> List[str]:
        return [k for k in COMPARE if k in self.bt.results]

    # ---------------------------------------------------------------------------
    # Curve di ricchezza cumulata
    # ---------------------------------------------------------------------------
    def equity_lines(self, interactive: bool = False):
        if interactive:
            return self._equity_plotly()
        fig, ax = plt.subplots(figsize=(10, 5.6))
        for k in self._keys():
            w = self._wealth(k)
            lw = 2.4 if k == "full" else 1.4
            ls = "--" if k == "equal_weight" else (":" if k == "bl_views" else "-")
            ax.plot(w.index, w.values, label=LABELS[k], color=PALETTE[k], lw=lw, ls=ls,
                    alpha=1.0 if k == "full" else 0.9)
        ax.set_ylabel("Ricchezza cumulata (base 100)")
        ax.set_title("Performance cumulata out-of-sample delle strategie a confronto")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left")
        fig.tight_layout()
        self._save(fig, "performance_cumulata")
        return fig

    def _equity_plotly(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        for k in self._keys():
            w = self._wealth(k)
            fig.add_trace(go.Scatter(
                x=w.index, y=w.values, name=LABELS[k], mode="lines",
                line=dict(color=PALETTE[k], width=3 if k == "full" else 1.6,
                          dash="dash" if k == "equal_weight" else ("dot" if k == "bl_views" else "solid")),
                hovertemplate="%{x|%b %Y}<br>%{y:.1f}<extra>" + LABELS[k] + "</extra>"))
        _layout(fig, "Performance cumulata out-of-sample", "Ricchezza cumulata (base 100)")
        fig.update_xaxes(rangeslider_visible=True)
        return fig

    # ---------------------------------------------------------------------------
    # Drawdown underwater
    # ---------------------------------------------------------------------------
    def drawdown(self, interactive: bool = False):
        if interactive:
            return self._drawdown_plotly()
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for k in self._keys():
            dd = self._drawdown(k)
            lw = 2.0 if k == "full" else 1.2
            ls = "--" if k == "equal_weight" else (":" if k == "bl_views" else "-")
            ax.plot(dd.index, dd.values, label=LABELS[k], color=PALETTE[k], lw=lw, ls=ls,
                    alpha=1.0 if k == "full" else 0.9)
        dd_full = self._drawdown("full")
        ax.fill_between(dd_full.index, dd_full.values, 0.0, color=PALETTE["full"], alpha=0.06)
        ax.axhline(0.0, color=PALETTE["ink"], lw=0.8)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown out-of-sample: profondità e durata delle perdite")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        self._save(fig, "drawdown_underwater")
        return fig

    def _drawdown_plotly(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        for k in self._keys():
            dd = self._drawdown(k)
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values, name=LABELS[k], mode="lines",
                fill="tozeroy" if k == "full" else None,
                fillcolor="rgba(17,17,17,0.06)",
                line=dict(color=PALETTE[k], width=3 if k == "full" else 1.4,
                          dash="dash" if k == "equal_weight" else ("dot" if k == "bl_views" else "solid")),
                hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>" + LABELS[k] + "</extra>"))
        _layout(fig, "Drawdown out-of-sample", "Drawdown (%)")
        return fig

    # ---------------------------------------------------------------------------
    # Dashboard combinato performance + drawdown (interattivo)
    # ---------------------------------------------------------------------------
    def performance_dashboard(self, interactive: bool = True):
        if not interactive:
            # versione statica a due pannelli
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                         gridspec_kw={"height_ratios": [2.4, 1]})
            for k in self._keys():
                w = self._wealth(k); dd = self._drawdown(k)
                lw = 2.2 if k == "full" else 1.3
                ls = "--" if k == "equal_weight" else (":" if k == "bl_views" else "-")
                a1.plot(w.index, w.values, color=PALETTE[k], lw=lw, ls=ls, label=LABELS[k])
                a2.plot(dd.index, dd.values, color=PALETTE[k], lw=lw, ls=ls)
            a1.set_ylabel("Ricchezza (base 100)"); a1.legend(loc="upper left")
            a1.set_title("Performance e drawdown out-of-sample")
            a2.set_ylabel("Drawdown (%)"); a2.axhline(0, color=PALETTE["ink"], lw=0.8)
            a2.xaxis.set_major_locator(mdates.YearLocator(2))
            a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.tight_layout()
            self._save(fig, "dashboard_performance")
            return fig
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                            vertical_spacing=0.06,
                            subplot_titles=("Ricchezza cumulata (base 100)", "Drawdown (%)"))
        for k in self._keys():
            w = self._wealth(k); dd = self._drawdown(k)
            dash = "dash" if k == "equal_weight" else ("dot" if k == "bl_views" else "solid")
            wd = 3 if k == "full" else 1.5
            fig.add_trace(go.Scatter(x=w.index, y=w.values, name=LABELS[k], legendgroup=k,
                                     line=dict(color=PALETTE[k], width=wd, dash=dash),
                                     hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>"+LABELS[k]+"</extra>"), 1, 1)
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=LABELS[k], legendgroup=k,
                                     showlegend=False,
                                     line=dict(color=PALETTE[k], width=wd, dash=dash),
                                     hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>"+LABELS[k]+"</extra>"), 2, 1)
        _layout(fig, "Performance e drawdown out-of-sample", None, height=650)
        return fig

    # ---------------------------------------------------------------------------
    # Scala di attribuzione con intervalli di confidenza
    # ---------------------------------------------------------------------------
    def attribution(self, interactive: bool = False):
        rf = self._rf()
        names, sr, ci = [], [], []
        step = {"markowitz": "Markowitz", "bl_classic": "+ Black-Litterman",
                "bl_views": "+ views LASSO", "full": "+ regimi (completo)"}
        for k in LADDER:
            if k not in self.bt.results:
                continue
            r = self.bt.results[k].returns
            s, se = _sharpe_se(r - rf.reindex(r.index).fillna(0.0))
            names.append(step[k]); sr.append(s); ci.append(1.96 * se)
        if interactive:
            return self._attribution_plotly(names, sr, ci)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        x = np.arange(len(names))
        colors = ["#B0B0B0"] * (len(names) - 1) + [PALETTE["full"]]
        ax.bar(x, sr, yerr=ci, capsize=5, color=colors, edgecolor=PALETTE["ink"],
               error_kw=dict(ecolor=PALETTE["ink"], lw=1.2))
        for i, s in enumerate(sr):
            ax.text(i, s + ci[i] + 0.03, f"{s:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Sharpe ratio (annualizzato)")
        ax.set_title("Scala di attribuzione: contributo di ciascun componente\n"
                     "(barre di errore = intervallo di confidenza 95%)")
        ax.grid(True, axis="y")
        fig.tight_layout()
        self._save(fig, "scala_attribuzione")
        return fig

    def _attribution_plotly(self, names, sr, ci):
        import plotly.graph_objects as go
        colors = ["#B0B0B0"] * (len(names) - 1) + [PALETTE["full"]]
        fig = go.Figure(go.Bar(
            x=names, y=sr, marker_color=colors,
            error_y=dict(type="data", array=ci, color=PALETTE["ink"], thickness=1.4, width=6),
            text=[f"{s:.2f}" for s in sr], textposition="outside",
            hovertemplate="%{x}<br>Sharpe %{y:.3f}<extra></extra>"))
        _layout(fig, "Scala di attribuzione (IC 95%)", "Sharpe ratio (annualizzato)")
        return fig

    # ---------------------------------------------------------------------------
    # Timeline dei regimi sul mercato
    # ---------------------------------------------------------------------------
    def regime_timeline(self, interactive: bool = False):
        mkt = self.md.benchmark.resample("ME").last()
        mkt = 100.0 * mkt / mkt.iloc[0]
        reg = self.rb.regimes.reindex(mkt.index)
        stress = reg.index[reg == "stress"]
        if interactive:
            return self._regime_plotly(mkt, stress)
        from matplotlib.patches import Patch
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(mkt.index, mkt.values, color=PALETTE["ink"], lw=1.6)
        for t in stress:
            ax.axvspan(t - pd.offsets.MonthBegin(1), t, color=PALETTE["stress"], alpha=0.22, lw=0)
        ax.set_ylabel("Indice di mercato (base 100)")
        ax.set_title("Regimi previsti dalla Random Forest e andamento del mercato")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(handles=[plt.Line2D([], [], color=PALETTE["ink"], lw=1.6, label="Mercato (ACWI)"),
                           Patch(facecolor=PALETTE["stress"], alpha=0.22, label="Mese previsto 'stress'")],
                  loc="upper left")
        fig.tight_layout()
        self._save(fig, "timeline_regimi")
        return fig

    def _regime_plotly(self, mkt, stress):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mkt.index, y=mkt.values, name="Mercato (ACWI)",
                                 line=dict(color=PALETTE["ink"], width=1.8),
                                 hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>"))
        for t in stress:
            fig.add_vrect(x0=t - pd.offsets.MonthBegin(1), x1=t,
                          fillcolor=PALETTE["stress"], opacity=0.20, line_width=0)
        _layout(fig, "Regimi previsti e andamento del mercato", "Indice di mercato (base 100)")
        return fig

    # ---------------------------------------------------------------------------
    # Heatmap delle allocazioni
    # ---------------------------------------------------------------------------
    def allocations_heatmap(self, key: str = "full", interactive: bool = False):
        W = self.bt.results[key].weights.copy()
        order = ([a.ticker for a in self.cfg.universe if a.asset_class == "safe"] +
                 [a.ticker for a in self.cfg.universe if a.asset_class == "risky"] +
                 [a.ticker for a in self.cfg.universe if a.asset_class == "real"])
        W = W.reindex(columns=[c for c in order if c in W.columns])
        if interactive:
            return self._heatmap_plotly(W)
        cmap = LinearSegmentedColormap.from_list("w", ["#FFFFFF", PALETTE["accent"]])
        cmap.set_bad("#EDEDED")
        fig, ax = plt.subplots(figsize=(11, 5.2))
        data = np.ma.masked_invalid(W.T.values)
        x = mdates.date2num(W.index.to_pydatetime())
        mesh = ax.pcolormesh(x, np.arange(len(W.columns)), data, cmap=cmap,
                             vmin=0, vmax=0.30, shading="nearest")
        ax.set_yticks(np.arange(len(W.columns))); ax.set_yticklabels(W.columns, fontsize=8)
        ax.invert_yaxis(); ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        cb = fig.colorbar(mesh, ax=ax, pad=0.01); cb.set_label("Peso in portafoglio")
        ax.set_title("Evoluzione delle allocazioni del framework completo")
        ax.grid(False)
        fig.tight_layout()
        self._save(fig, "heatmap_allocazioni")
        return fig

    def _heatmap_plotly(self, W):
        import plotly.graph_objects as go
        fig = go.Figure(go.Heatmap(
            z=W.T.values, x=W.index, y=W.columns,
            colorscale=[[0, "#FFFFFF"], [1, PALETTE["accent"]]], zmin=0, zmax=0.30,
            colorbar=dict(title="Peso"), hoverongaps=False,
            hovertemplate="%{y}<br>%{x|%b %Y}: %{z:.1%}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        _layout(fig, "Evoluzione delle allocazioni del framework completo", None)
        return fig

    # ---------------------------------------------------------------------------
    # Correlazione media rolling degli asset rischiosi (diversificazione)
    # ---------------------------------------------------------------------------
    def rolling_correlation(self, window: int = 12, interactive: bool = False):
        """Correlazione media a coppie tra gli asset rischiosi, su finestra mobile.
        Sovrappone i mesi 'stress': mostra che in crisi le correlazioni salgono e
        la diversificazione si contrae — la motivazione empirica del de-risking."""
        risky = [a.ticker for a in self.cfg.universe if a.asset_class == "risky"]
        simple = self.md.prices_eur.resample("ME").last().pct_change()
        R = simple[[c for c in risky if c in simple.columns]]
        # media delle correlazioni a coppie su ogni finestra
        avg_corr = {}
        idx = R.index
        for i in range(window, len(idx) + 1):
            sub = R.iloc[i - window:i].dropna(axis=1, how="any")
            if sub.shape[1] < 2:
                continue
            C = np.corrcoef(sub.values.T)
            iu = np.triu_indices_from(C, k=1)
            avg_corr[idx[i - 1]] = float(np.nanmean(C[iu]))
        ac = pd.Series(avg_corr).sort_index()
        stress = self.rb.regimes.reindex(ac.index)
        stress_dates = stress.index[stress == "stress"]
        if interactive:
            return self._corr_plotly(ac, stress_dates, window)
        from matplotlib.patches import Patch
        fig, ax = plt.subplots(figsize=(10, 4.6))
        for t in stress_dates:
            ax.axvspan(t - pd.offsets.MonthBegin(1), t, color=PALETTE["stress"], alpha=0.20, lw=0)
        ax.plot(ac.index, ac.values, color=PALETTE["accent"], lw=1.8)
        ax.axhline(ac.mean(), color=PALETTE["ink"], lw=0.9, ls=":", label=f"media {ac.mean():.2f}")
        ax.set_ylabel("Correlazione media a coppie")
        ax.set_title(f"Correlazione media degli asset rischiosi (finestra {window}m)")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(handles=[plt.Line2D([], [], color=PALETTE["accent"], lw=1.8, label="Correlazione media"),
                           plt.Line2D([], [], color=PALETTE["ink"], lw=0.9, ls=":", label=f"media periodo {ac.mean():.2f}"),
                           Patch(facecolor=PALETTE["stress"], alpha=0.20, label="Mese 'stress'")],
                  loc="lower right")
        fig.tight_layout()
        self._save(fig, "rolling_correlation")
        return fig

    def _corr_plotly(self, ac, stress_dates, window):
        import plotly.graph_objects as go
        fig = go.Figure()
        for t in stress_dates:
            fig.add_vrect(x0=t - pd.offsets.MonthBegin(1), x1=t,
                          fillcolor=PALETTE["stress"], opacity=0.18, line_width=0)
        fig.add_trace(go.Scatter(x=ac.index, y=ac.values, name="Correlazione media",
                                 line=dict(color=PALETTE["accent"], width=2),
                                 hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=float(ac.mean()), line=dict(color=PALETTE["ink"], dash="dot", width=1))
        _layout(fig, f"Correlazione media degli asset rischiosi (finestra {window}m)",
                "Correlazione media a coppie")
        return fig

    # ---------------------------------------------------------------------------
    # Importanza delle feature
    # ---------------------------------------------------------------------------
    def feature_importance(self, top: int = 10, interactive: bool = False):
        imp = self.rb.importances.head(top).iloc[::-1]
        if interactive:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                                   marker_color=PALETTE["markowitz"],
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            _layout(fig, "Feature più informative per i regimi", "Importanza media")
            return fig
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.barh(range(len(imp)), imp.values, color=PALETTE["markowitz"],
                edgecolor=PALETTE["ink"], height=0.7)
        ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=9)
        ax.set_xlabel("Importanza media (riduzione di impurità)")
        ax.set_title("Feature più informative per il rilevamento dei regimi")
        ax.grid(True, axis="x")
        fig.tight_layout()
        self._save(fig, "feature_importance")
        return fig

    # ---------------------------------------------------------------------------
    # Scheda di portafoglio: ultima allocazione + metriche complete
    # ---------------------------------------------------------------------------
    def _class_of(self) -> Dict[str, str]:
        return {a.ticker: a.asset_class for a in self.cfg.universe}

    def snapshot_table(self, key: str = "full", capital: float = 1_000_000.0) -> pd.DataFrame:
        """Tabella semplice (Metrica | Valore) con tutti gli indicatori della scheda."""
        m = self.snapshot_metrics(key, capital)
        eur = lambda v: f"€{v:,.0f}".replace(",", ".")
        pct = lambda v: f"{v*100:.2f}%"
        spct = lambda v: f"{v*100:+.2f}%"
        rows = [
            ("Capitale iniziale", eur(m["capital"])),
            (f"Valore al {m['last_date']:%d/%m/%Y}", eur(m["final_value"])),
            ("P&L assoluto", eur(m["pnl_abs"])),
            ("P&L percentuale", spct(m["pnl_pct"])),
            ("CAGR", pct(m["cagr"])),
            ("Volatilità annua", pct(m["ann_vol"])),
            ("Volatilità giornaliera", pct(m["daily_vol"])),
            ("Max drawdown", pct(m["max_dd"])),
            ("Sharpe ratio", f"{m['sharpe']:.2f}"),
            ("Sortino ratio", f"{m['sortino']:.2f}"),
            ("Calmar ratio", f"{m['calmar']:.2f}"),
            ("Information ratio", f"{m['info_ratio']:.2f}"),
            ("Tracking error (TEV)", pct(m["tev"])),
            ("CEQ (certainty equivalent)", pct(m["ceq"])),
            ("Win rate", f"{m['win_rate']*100:.1f}%"),
            ("Best month", f"{spct(m['best_month'])} ({m['best_month_date']:%b %Y})"),
            ("Worst month", f"{spct(m['worst_month'])} ({m['worst_month_date']:%b %Y})"),
            ("Best asset (contributo)", f"{m['best_asset']} ({spct(m['best_asset_val'])})"),
            ("Worst asset (contributo)", f"{m['worst_asset']} ({spct(m['worst_asset_val'])})"),
            ("Mesi di backtest", f"{m['n_months']}"),
        ]
        return pd.DataFrame(rows, columns=["Metrica", "Valore"])

    def _daily_vol(self, key: str) -> float:
        """Volatilità giornaliera del portafoglio, ricostruita tenendo i pesi
        decisi a fine mese t costanti sui giorni del mese t+1 (no drift infra-mese)."""
        if self.md is None or self.md.prices_eur.empty:
            return float("nan")
        W = self.bt.results[key].weights
        daily = self.md.prices_eur.pct_change()
        parts = []
        for t in W.index:
            w = W.loc[t].fillna(0.0)
            days = daily.loc[(daily.index > t) & (daily.index <= t + pd.offsets.MonthEnd(1))]
            if days.empty:
                continue
            sub = days.reindex(columns=w.index).fillna(0.0)
            parts.append(sub.mul(w, axis=1).sum(axis=1).to_numpy())
        if not parts:
            return float("nan")
        alld = np.concatenate(parts)
        return float(np.std(alld, ddof=1))

    def _contributions(self, key: str) -> pd.Series:
        """Contributo di ciascun ETF al rendimento del portafoglio: Σ_t w[t]·r[t+1]."""
        simple = self.md.prices_eur.resample("ME").last().pct_change()
        W = self.bt.results[key].weights
        contrib = pd.Series(0.0, index=W.columns)
        months = list(simple.index)
        for t in W.index:
            if t not in simple.index:
                continue
            pos = months.index(t)
            if pos + 1 >= len(months):
                continue
            r_next = simple.loc[months[pos + 1]].reindex(W.columns)
            w = W.loc[t].reindex(W.columns).fillna(0.0)
            contrib = contrib.add((w * r_next).fillna(0.0), fill_value=0.0)
        return contrib

    def snapshot_metrics(self, key: str = "full", capital: float = 1_000_000.0) -> Dict:
        """Tutte le metriche della scheda, in un dizionario (testabile e riusabile)."""
        res = self.bt.results[key]
        r = res.returns.sort_index()
        n = len(r)
        years = n / _MPY
        rf = self._rf().reindex(r.index).fillna(0.0)
        excess = r - rf

        wealth = capital * (1.0 + r).cumprod()
        final_value = float(wealth.iloc[-1])
        max_dd = float((wealth / wealth.cummax() - 1.0).min())

        # rischio
        ann_vol = float(r.std(ddof=1)) * np.sqrt(_MPY)
        downside = np.minimum(excess.to_numpy(), 0.0)
        dd_dev = float(np.sqrt((downside ** 2).mean()))
        # benchmark per IR/TEV
        bench = self.md.benchmark.resample("ME").last().pct_change().reindex(r.index)
        active = (r - bench).dropna()
        tev = float(active.std(ddof=1)) * np.sqrt(_MPY) if len(active) > 1 else float("nan")

        cagr = (final_value / capital) ** (1.0 / years) - 1.0
        contrib = self._contributions(key)

        def _ratio(num, den):
            return float(num / den) if den and np.isfinite(den) and den != 0 else float("nan")

        return {
            "last_date": res.weights.index[-1],
            "capital": capital,
            "final_value": final_value,
            "pnl_abs": final_value - capital,
            "pnl_pct": final_value / capital - 1.0,
            "cagr": cagr,
            "sharpe": res.metrics.get("sharpe", float("nan")),
            # annualizzati come lo Sharpe: rendimento ×12, rischio ×√12
            "sortino": _ratio(excess.mean() * _MPY, dd_dev * np.sqrt(_MPY)),
            "calmar": _ratio(cagr, abs(max_dd)),
            "info_ratio": _ratio(active.mean() * _MPY, tev) if len(active) > 1 else float("nan"),
            "win_rate": float((r > 0).mean()),
            "best_month": float(r.max()), "best_month_date": r.idxmax(),
            "worst_month": float(r.min()), "worst_month_date": r.idxmin(),
            "ann_vol": ann_vol,
            "daily_vol": self._daily_vol(key),
            "max_dd": max_dd,
            "tev": tev,
            "ceq": res.metrics.get("ceq", float("nan")),
            "best_asset": contrib.idxmax(), "best_asset_val": float(contrib.max()),
            "worst_asset": contrib.idxmin(), "worst_asset_val": float(contrib.min()),
            "n_months": n,
        }

    def allocation_snapshot(self, key: str = "full", capital: float = 1_000_000.0,
                            interactive: bool = False):
        res = self.bt.results[key]
        w_last = res.weights.iloc[-1].dropna()
        w_last = w_last[w_last > 1e-6].sort_values(ascending=False)
        cls = self._class_of()
        m = self.snapshot_metrics(key, capital)
        # aggregato per classe
        by_class = w_last.groupby(w_last.index.map(cls)).sum()
        by_class = by_class.reindex([c for c in ("safe", "risky", "real") if c in by_class.index])
        if interactive:
            return self._snapshot_plotly(w_last, by_class, cls, m)

        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(2, 12, height_ratios=[1.05, 1.0], hspace=0.35, wspace=0.6)

        # --- torta per classe ---
        ax_pie = fig.add_subplot(gs[0, 0:5])
        ax_pie.pie(by_class.values, labels=[CLASS_LABELS[c] for c in by_class.index],
                   colors=[CLASS_COLORS[c] for c in by_class.index],
                   autopct=lambda p: f"{p:.0f}%", startangle=90,
                   wedgeprops=dict(edgecolor="white", linewidth=1.5),
                   textprops=dict(fontsize=10))
        ax_pie.set_title("Allocazione per classe")

        # --- barre per ETF ---
        ax_bar = fig.add_subplot(gs[0, 5:12])
        y = np.arange(len(w_last))[::-1]
        colors = [CLASS_COLORS[cls.get(t, "risky")] for t in w_last.index]
        ax_bar.barh(y, w_last.values * 100, color=colors, edgecolor=PALETTE["ink"], height=0.72)
        ax_bar.set_yticks(y); ax_bar.set_yticklabels(w_last.index, fontsize=9)
        for yi, v in zip(y, w_last.values * 100):
            ax_bar.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=8)
        ax_bar.set_xlabel("Peso (%)")
        ax_bar.set_title(f"Ultima allocazione disponibile ({m['last_date']:%b %Y})")
        ax_bar.grid(True, axis="x")

        # --- pannelli metriche ---
        def panel(col0, col1, title, rows):
            ax = fig.add_subplot(gs[1, col0:col1]); ax.axis("off")
            ax.text(0, 1.0, title, fontsize=11, fontweight="semibold",
                    color=PALETTE["accent"], transform=ax.transAxes)
            for i, (lab, val) in enumerate(rows):
                yy = 0.86 - i * 0.115
                ax.text(0.0, yy, lab, fontsize=9.5, transform=ax.transAxes)
                ax.text(1.0, yy, val, fontsize=9.5, fontweight="semibold", ha="right",
                        transform=ax.transAxes)

        eur = lambda v: f"€{v:,.0f}".replace(",", ".")
        pct = lambda v: f"{v*100:+.2f}%"
        panel(0, 4, "Valore & P&L", [
            ("Capitale iniziale", eur(m["capital"])),
            (f"Valore al {m['last_date']:%d/%m/%Y}", eur(m["final_value"])),
            ("P&L assoluto", eur(m["pnl_abs"])),
            ("P&L percentuale", pct(m["pnl_pct"])),
            ("CAGR", f"{m['cagr']*100:.2f}%"),
            ("Win rate", f"{m['win_rate']*100:.1f}%"),
        ])
        panel(4, 8, "Performance", [
            ("Sharpe", f"{m['sharpe']:.2f}"),
            ("Sortino", f"{m['sortino']:.2f}"),
            ("Calmar", f"{m['calmar']:.2f}"),
            ("Information ratio", f"{m['info_ratio']:.2f}"),
            ("Best month", f"{pct(m['best_month'])} ({m['best_month_date']:%b %y})"),
            ("Worst month", f"{pct(m['worst_month'])} ({m['worst_month_date']:%b %y})"),
        ])
        panel(8, 12, "Rischio & attribuzione", [
            ("Volatilità annua", f"{m['ann_vol']*100:.2f}%"),
            ("Volatilità giornaliera", f"{m['daily_vol']*100:.2f}%"),
            ("Max drawdown", f"{m['max_dd']*100:.2f}%"),
            ("Tracking error (TEV)", f"{m['tev']*100:.2f}%"),
            ("Best asset", f"{m['best_asset']} ({m['best_asset_val']*100:+.1f}%)"),
            ("Worst asset", f"{m['worst_asset']} ({m['worst_asset_val']*100:+.1f}%)"),
        ])

        fig.suptitle("Scheda di portafoglio — framework completo", fontsize=15,
                     fontweight="semibold", y=0.98)
        self._save(fig, "scheda_portafoglio")
        return fig

    def _snapshot_plotly(self, w_last, by_class, cls, m):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        eur = lambda v: f"€{v:,.0f}".replace(",", ".")
        fig = make_subplots(
            rows=2, cols=2, row_heights=[0.5, 0.5],
            specs=[[{"type": "domain"}, {"type": "xy"}],
                   [{"type": "table", "colspan": 2}, None]],
            subplot_titles=("Allocazione per classe",
                            f"Ultima allocazione ({m['last_date']:%b %Y})", ""))
        fig.add_trace(go.Pie(labels=[CLASS_LABELS[c] for c in by_class.index],
                             values=by_class.values,
                             marker_colors=[CLASS_COLORS[c] for c in by_class.index],
                             hole=0.4, textinfo="label+percent"), 1, 1)
        fig.add_trace(go.Bar(x=w_last.values[::-1] * 100, y=w_last.index[::-1],
                             orientation="h",
                             marker_color=[CLASS_COLORS[cls.get(t, "risky")] for t in w_last.index[::-1]],
                             hovertemplate="%{y}: %{x:.1f}%<extra></extra>"), 1, 2)
        rows = [
            ("Capitale iniziale", eur(m["capital"])),
            (f"Valore al {m['last_date']:%d/%m/%Y}", eur(m["final_value"])),
            ("P&L assoluto", eur(m["pnl_abs"])),
            ("P&L %", f"{m['pnl_pct']*100:+.2f}%"),
            ("CAGR", f"{m['cagr']*100:.2f}%"),
            ("Sharpe", f"{m['sharpe']:.2f}"),
            ("Sortino", f"{m['sortino']:.2f}"),
            ("Calmar", f"{m['calmar']:.2f}"),
            ("Information ratio", f"{m['info_ratio']:.2f}"),
            ("Win rate", f"{m['win_rate']*100:.1f}%"),
            ("Best month", f"{m['best_month']*100:+.2f}% ({m['best_month_date']:%b %y})"),
            ("Worst month", f"{m['worst_month']*100:+.2f}% ({m['worst_month_date']:%b %y})"),
            ("Volatilità annua", f"{m['ann_vol']*100:.2f}%"),
            ("Volatilità giornaliera", f"{m['daily_vol']*100:.2f}%"),
            ("Max drawdown", f"{m['max_dd']*100:.2f}%"),
            ("Tracking error (TEV)", f"{m['tev']*100:.2f}%"),
            ("CEQ", f"{m['ceq']*100:.2f}%"),
            ("Best asset", f"{m['best_asset']} ({m['best_asset_val']*100:+.1f}%)"),
            ("Worst asset", f"{m['worst_asset']} ({m['worst_asset_val']*100:+.1f}%)"),
        ]
        fig.add_trace(go.Table(
            header=dict(values=["<b>Metrica</b>", "<b>Valore</b>"],
                        fill_color=PALETTE["accent"], font=dict(color="white", size=12),
                        align="left"),
            cells=dict(values=[[a for a, _ in rows], [b for _, b in rows]],
                       align="left", height=24,
                       fill_color=[["#F5F5F5", "white"] * 10])), 2, 1)
        fig.update_layout(template="plotly_white", height=820, showlegend=False,
                          title=dict(text="Scheda di portafoglio — framework completo",
                                     font=dict(size=17, color=PALETTE["ink"])),
                          margin=dict(l=40, r=30, t=70, b=30))
        return fig

    # ---------------------------------------------------------------------------
    # Metriche rolling (non-stazionarieta' + tenuta del target-vol)
    # ---------------------------------------------------------------------------
    def rolling_metrics(self, window: int = 36, keys=None, interactive: bool = False):
        keys = keys or [k for k in ("markowitz", "full", "equal_weight") if k in self.bt.results]
        rf = self._rf()
        roll_sr, roll_vol = {}, {}
        for k in keys:
            r = self.bt.results[k].returns.sort_index()
            ex = r - rf.reindex(r.index).fillna(0.0)
            mu = ex.rolling(window).mean()
            sd = r.rolling(window).std(ddof=1)
            roll_sr[k] = (mu / r.rolling(window).std(ddof=1) * np.sqrt(_MPY)).dropna()
            roll_vol[k] = (sd * np.sqrt(_MPY) * 100).dropna()
        tgt = self.cfg.risk.target_vol_annual * 100
        if interactive:
            return self._rolling_plotly(roll_sr, roll_vol, tgt, window)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1]})
        for k in keys:
            a1.plot(roll_sr[k].index, roll_sr[k].values, color=PALETTE[k],
                    lw=2.2 if k == "full" else 1.4, label=LABELS[k])
            a2.plot(roll_vol[k].index, roll_vol[k].values, color=PALETTE[k],
                    lw=2.2 if k == "full" else 1.4, label=LABELS[k])
        a1.axhline(0, color=PALETTE["ink"], lw=0.8)
        a1.set_ylabel(f"Sharpe rolling ({window}m)"); a1.legend(loc="upper left")
        a1.set_title(f"Metriche su finestra mobile di {window} mesi")
        a2.axhline(tgt, color=PALETTE["stress"], lw=1.3, ls="--", label=f"target {tgt:.0f}%")
        a2.set_ylabel(f"Volatilità ann. rolling (%)"); a2.legend(loc="upper left")
        a2.xaxis.set_major_locator(mdates.YearLocator(2))
        a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        self._save(fig, "rolling_metrics")
        return fig

    def _rolling_plotly(self, roll_sr, roll_vol, tgt, window):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=(f"Sharpe rolling ({window}m)",
                                            f"Volatilità annualizzata rolling ({window}m)"))
        for k in roll_sr:
            w = 3 if k == "full" else 1.6
            fig.add_trace(go.Scatter(x=roll_sr[k].index, y=roll_sr[k].values, name=LABELS[k],
                          legendgroup=k, line=dict(color=PALETTE[k], width=w),
                          hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>"+LABELS[k]+"</extra>"), 1, 1)
            fig.add_trace(go.Scatter(x=roll_vol[k].index, y=roll_vol[k].values, name=LABELS[k],
                          legendgroup=k, showlegend=False, line=dict(color=PALETTE[k], width=w),
                          hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>"+LABELS[k]+"</extra>"), 2, 1)
        fig.add_hline(y=tgt, line=dict(color=PALETTE["stress"], dash="dash", width=1.3), row=2, col=1)
        _layout(fig, f"Metriche su finestra mobile di {window} mesi", None, height=650)
        return fig

    # ---------------------------------------------------------------------------
    # Contributo al rischio per asset (budget di rischio vs peso)
    # ---------------------------------------------------------------------------
    def risk_contribution(self, key: str = "full", window: int = 60, interactive: bool = False):
        from sklearn.covariance import LedoitWolf
        W = self.bt.results[key].weights
        w_last = W.iloc[-1].dropna()
        w_last = w_last[w_last > 1e-6]
        assets = list(w_last.index)
        t_last = W.index[-1]
        # Sigma stimata (Ledoit-Wolf) sui rendimenti realizzati fino a t_last
        simple = self.md.prices_eur.resample("ME").last().pct_change()
        hist = simple.loc[simple.index <= t_last, assets].dropna().iloc[-window:]
        S = LedoitWolf().fit(hist.values).covariance_
        w = w_last.values
        port_var = float(w @ S @ w)
        mrc = S @ w                                  # contributo marginale
        rc = w * mrc / port_var                      # contributo al rischio (somma 1)
        df = pd.DataFrame({"peso": w, "contributo": rc}, index=assets).sort_values("contributo")
        cls = self._class_of()
        if interactive:
            return self._rc_plotly(df, cls, key)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        y = np.arange(len(df)); h = 0.38
        ax.barh(y + h/2, df["peso"].values*100, height=h, color="#B0B0B0",
                edgecolor=PALETTE["ink"], label="Peso")
        ax.barh(y - h/2, df["contributo"].values*100, height=h,
                color=[CLASS_COLORS.get(cls.get(a, "risky"), "#888") for a in df.index],
                edgecolor=PALETTE["ink"], label="Contributo al rischio")
        ax.set_yticks(y); ax.set_yticklabels(df.index, fontsize=8)
        ax.set_xlabel("% del portafoglio / del rischio")
        ax.set_title(f"Peso vs contributo al rischio — {LABELS.get(key, key)} ({t_last:%b %Y})")
        ax.legend(loc="lower right"); ax.grid(True, axis="x")
        fig.tight_layout()
        self._save(fig, "risk_contribution")
        return fig

    def _rc_plotly(self, df, cls, key):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df.index, x=df["peso"]*100, orientation="h", name="Peso",
                             marker_color="#B0B0B0",
                             hovertemplate="%{y}: %{x:.1f}%<extra>peso</extra>"))
        fig.add_trace(go.Bar(y=df.index, x=df["contributo"]*100, orientation="h",
                             name="Contributo al rischio",
                             marker_color=[CLASS_COLORS.get(cls.get(a, "risky"), "#888") for a in df.index],
                             hovertemplate="%{y}: %{x:.1f}%<extra>rischio</extra>"))
        fig.update_layout(barmode="group")
        _layout(fig, f"Peso vs contributo al rischio — {LABELS.get(key, key)}", "% del portafoglio / del rischio")
        return fig

    # ---- comodo: genera e salva tutti i grafici statici -------------------
    def save_all(self) -> None:
        self.equity_lines(); self.drawdown(); self.attribution()
        self.regime_timeline(); self.allocations_heatmap(); self.feature_importance()
        self.allocation_snapshot()
        self.rolling_metrics(); self.risk_contribution(); self.rolling_correlation()
        plt.close("all")

def _layout(fig, title: str, ytitle: Optional[str], height: int = 480) -> None:
    """Layout plotly coerente col tema statico."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=PALETTE["ink"])),
        template="plotly_white", height=height,
        font=dict(family="DejaVu Sans, Arial", size=12, color=PALETTE["ink"]),
        legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
        margin=dict(l=60, r=30, t=60, b=40), hovermode="x unified")
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"])