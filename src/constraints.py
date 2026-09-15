# =============================================================================
# CONSTRAINTS.PY — Blocco 6 · Vincoli di portafoglio per regime
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import Config, CONFIG

logger = logging.getLogger(__name__)
_TOL = 1e-6      # tolleranza numerica
_GROUPS = ("defensive", "govt", "cash", "equity", "credit", "gold")

# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------
@dataclass
class GroupConstraint:
    # Vincolo aggregato su un gruppo di asset
    name: str
    members: List[int]       
    lower: Optional[float]
    upper: Optional[float]

@dataclass
class ConstraintSet:
    # Insieme dei vincoli per un singolo mese
    assets: List[str]
    regime: str
    lb: np.ndarray                  # bound inferiori per-asset
    ub: np.ndarray                  # bound superiori per-asset
    groups: List[GroupConstraint]
    fully_invested: bool
    long_only: bool
    feasible: bool
    reason: Optional[str] = None
    relaxations: List[str] = field(default_factory=list)  # floor di gruppo rilassati

    def check_weights(self, w, tol: float = 1e-4) -> List[str]:
        # Validatore a posteriori
        w = np.asarray(w, dtype=float)
        v: List[str] = []
        if self.long_only and (w < -tol).any():
            v.append("pesi negativi (viola long-only)")
        if (w > self.ub + tol).any():
            v.append("single_asset_cap superato")
        if self.fully_invested and abs(w.sum() - 1.0) > 1e-3:
            v.append(f"somma pesi {w.sum():.4f} != 1")
        for g in self.groups:
            s = float(w[g.members].sum())
            if g.upper is not None and s > g.upper + tol:
                v.append(f"gruppo '{g.name}' {s:.3f} > upper {g.upper}")
            if g.lower is not None and s < g.lower - tol:
                v.append(f"gruppo '{g.name}' {s:.3f} < lower {g.lower}")
        return v

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class ConstraintBuilder:
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.cc = config.constraints
        # Mappa gruppo -> ticker
        self._group_members: Dict[str, set] = {
            g: set(config.tickers_in_group(g)) for g in _GROUPS}
        # Mappa ticker -> sotto-gruppo
        self._sub_group: Dict[str, str] = {a.ticker: a.sub_group for a in config.universe}

    def _regime_key(self, regime: str) -> str:
        # Normalizza il regime in caso di input non previsti
        return regime if regime in self.cc.single_asset_cap else "normal"

    # ---- Costruzione ------------------------------------------------------
    def build(self, assets: List[str], regime: str = "normal",
              relax: bool = True) -> ConstraintSet:
        rk = self._regime_key(regime)
        n = len(assets)
        cap = float(self.cc.single_asset_cap[rk])
        override = self.cc.asset_cap_override.get(rk, {})
        lb = np.zeros(n) if self.cc.long_only else np.full(n, -np.inf)
        # Cap per-asset con deroga per sotto-gruppo
        ub = np.array([override.get(self._sub_group.get(a), cap) for a in assets], dtype=float)
        groups: List[GroupConstraint] = []
        relaxations: List[str] = []
        # Costruisce i vincoli di gruppo sull'universo effettivamente presente
        for gname, (glo, ghi) in self.cc.group_limits[rk].items():
            members = [i for i, a in enumerate(assets)
                       if a in self._group_members.get(gname, set())]
            if not members:
                continue
            eff_lo = glo
            if relax and glo is not None:
                max_reach = float(ub[members].sum())
                if max_reach < glo - _TOL:
                    eff_lo = round(max_reach, 6)
                    relaxations.append(
                        f"{gname}: floor {glo:.2f} -> {eff_lo:.2f} ({len(members)} asset, max {max_reach:.2f})")
            groups.append(GroupConstraint(gname, members, eff_lo, ghi))
        feasible, reason = self._feasibility(assets, ub, groups, rk)
        cs = ConstraintSet(assets=list(assets), regime=rk, lb=lb, ub=ub, groups=groups,
                           fully_invested=self.cc.fully_invested,
                           long_only=self.cc.long_only, feasible=feasible, reason=reason,
                           relaxations=relaxations)
        if relaxations:
            logger.info("Vincoli rilassati (regime=%s, %d asset): %s", rk, n, "; ".join(relaxations))
        if not feasible:
            logger.warning("Vincoli infeasible (regime=%s, %d asset): %s", rk, n, reason)
        return cs

    # ---- Fattibilità ------------------------------------------------------
    def _feasibility(self, assets: List[str], ub: np.ndarray,
                     groups: List[GroupConstraint],
                     rk: str) -> Tuple[bool, Optional[str]]:
        # Verifica le condizioni necessarie perche' esista un peso ammissibile
        n = len(assets)
        reasons: List[str] = []
        if self.cc.fully_invested:
            # Condizione 1) i cap devono poter sommare ad 1
            if ub.sum() < 1.0 - _TOL:
                reasons.append(f"somma dei cap ({ub.sum():.2f}) < 1: impossibile essere fully invested")
            # Condizione 2) ogni lower di gruppo dev'essere raggiungibile coi cap dei suoi membri
            for g in groups:
                if g.lower is not None:
                    max_reach = float(ub[g.members].sum())
                    if max_reach < g.lower - _TOL:
                        reasons.append(
                            f"gruppo '{g.name}': massimo raggiungibile {max_reach:.2f} < lower {g.lower:.2f} "
                            f"({len(g.members)} asset)")
            # Condizione 3) la somma dei lower di gruppo non puo' eccedere 1
            low_sum = sum(g.lower for g in groups if g.lower is not None)
            if low_sum > 1.0 + _TOL:
                reasons.append(f"somma dei lower di gruppo ({low_sum:.2f}) > 1")
            # Condizione 4) gli upper di gruppo devono lasciare spazio per arrivare a 1
            covered = set()
            upper_budget = 0.0
            capped_all = True
            for g in groups:
                if g.upper is not None:
                    upper_budget += g.upper
                    covered.update(g.members)
            if len(covered) == n and capped_all and upper_budget < 1.0 - _TOL:
                reasons.append(f"somma degli upper di gruppo ({upper_budget:.2f}) < 1 con tutti gli asset vincolati")
        return (len(reasons) == 0), ("; ".join(reasons) if reasons else None)