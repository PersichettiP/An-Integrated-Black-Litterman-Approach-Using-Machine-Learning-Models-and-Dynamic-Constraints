# =============================================================================
# CONFIG.PY — Configurazione centralizzata del framework
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Riproducibilità e costanti globali
# -----------------------------------------------------------------------------
SEED: int = 42
TRADING_DAYS_PER_YEAR: int = 252
MONTHS_PER_YEAR: int = 12

# Directory di lavoro
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()

# -----------------------------------------------------------------------------
# Universo investibile
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Asset:
    ticker: str          # identificativo univoco
    filename: str        # file Excel sorgente
    name: str            # denominazione estesa
    asset_class: str     # macro-categoria funzionale
    sub_group: str       # sottoclasse
    quote_ccy: str       # valuta di denominazione

ASSET_UNIVERSE: Tuple[Asset, ...] = (
    # ---- Safe assets --------------------------------------------------------
    Asset("XEON.DE", "XEON.DE.xlsx",
          "Xtrackers II EUR Overnight Rate Swap UCITS 1C", "safe", "cash", "EUR"),
    Asset("DBXP.DE", "DBXP.DE.xlsx",
          "Xtrackers II Eurozone Gov Bond 1-3 UCITS 1C", "safe", "govt", "EUR"),
    Asset("SYBA.DE", "SYBA.DE.xlsx",
          "SPDR Bloomberg Euro Aggregate Bond UCITS", "safe", "govt", "EUR"),
    Asset("SYBZ.DE", "SYBZ.DE.xlsx",
          "SPDR Bloomberg Global Aggregate Bond UCITS", "safe", "govt", "USD"),
    # ---- Risky assets -------------------------------------------------------
    Asset("IHYG.L", "IHYG.L.xlsx",
          "iShares EUR High Yield Corp Bond UCITS", "risky", "credit_hy", "EUR"),
    Asset("IEMB.L", "IEMB.L.xlsx",
          "iShares JP Morgan USD EM Bond UCITS", "risky", "em_bond", "USD"),
    Asset("SMEA.L", "SMEA.L.xlsx",
          "iShares Core MSCI Europe UCITS", "risky", "equity", "EUR"),
    Asset("IQQN.DE", "IQQN.DE.xlsx",
          "iShares MSCI North America UCITS", "risky", "equity", "USD"),
    Asset("SXR1.DE", "SXR1.DE.xlsx",
          "iShares Core MSCI Pacific ex-Japan UCITS", "risky", "equity", "USD"),
    Asset("SJPA.L", "SJPA.L.xlsx",
          "iShares Core MSCI Japan IMI UCITS", "risky", "equity", "USD"),
    Asset("IQQE.DE", "IQQE.DE.xlsx",
          "iShares MSCI EM UCITS", "risky", "equity", "USD"),
    # ---- Real assets --------------------------------------------------------
    Asset("IGLN.L", "IGLN.L.xlsx",
          "iShares Physical Gold ETC", "real", "gold", "USD"),
)

# Proxy del tasso risk-free
RISK_FREE_TICKER: str = "XEON.DE"

# -----------------------------------------------------------------------------
# Dati e percorsi
# -----------------------------------------------------------------------------
@dataclass
class DataConfig:
    data_dir: Path = _HERE
    base_ccy: str = "EUR"                 # valuta di riferimento del portafoglio
    fred_dir: Path = _HERE
    fred_file_ext: str = "xlsx"
    # Cambi diretti via FRED, convenzione "valuta estera per 1 EUR"
    fx_direct: Dict[str, str] = field(default_factory=lambda: {
        "USD": "DEXUSEU",
    })
    # Benchmark del portafoglio di mercato
    benchmark_file: str = "SSAC.L.xlsx"   # iShares MSCI ACWI UCITS USD Acc
    benchmark_ticker: str = "ACWI"
    benchmark_ccy: str = "USD"
    # Variabili macro-finanziarie
    macro_files: Dict[str, str] = field(default_factory=lambda: {
        "VIX": "VOV.xlsx",                  # CBOE Volatility Index (+ Vol of Vol)
        "HY_OAS": "CREDITSPREAD.xlsx",      # High Yield Option-Adjusted Spread
        "NFCI": "LIQUIDITY.xlsx",           # National Financial Conditions Index
        "SLOPE": "SLOPE.xlsx",              # Pendenza 10Y-2Y Treasury
        "REAL_RATE": "10YTREASURY.xlsx",    # 10Y yield / breakeven / tasso reale
        "BREAKEVEN": "INFLATION.xlsx",      # 10Y breakeven inflation
        "DXY": "DOLLAR.xlsx",               # US Dollar Index
        "PPI": "COMMODITIES.xlsx",          # Producer Price Index
    })
    # Orizzonte temporale dell'analisi
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    max_ffill_days: int = 5               # tolleranza sui micro-gap di calendario

# -----------------------------------------------------------------------------
# Black–Litterman
# -----------------------------------------------------------------------------
@dataclass
class BlackLittermanConfig:
    tau: float = 0.05                     # incertezza standard del prior
    delta_mode: str = "fixed"
    delta_fixed: float = 4.5
    # Pesi strategici del portafoglio di riferimento (policy prior)
    strategic_weights: Dict[str, float] = field(default_factory=lambda: {
        # Safe 40%
        "XEON.DE": 0.10, "DBXP.DE": 0.10, "SYBA.DE": 0.10, "SYBZ.DE": 0.10,
        # Risky 50% (equity 38% + credito 12%)
        "SMEA.L": 0.10, "IQQN.DE": 0.12, "SXR1.DE": 0.04,
        "SJPA.L": 0.04, "IQQE.DE": 0.08,
        "IHYG.L": 0.06, "IEMB.L": 0.06,
        # Real 10%
        "IGLN.L": 0.10,
    })

# -----------------------------------------------------------------------------
# Views via LASSO
# -----------------------------------------------------------------------------
@dataclass
class LassoConfig:
    train_window_months: int = 60         # finestra di stima rolling
    min_train_months: int = 36            # storia minima per iniziare il backtest
    n_lambda: int = 100                   # griglia di lambda per la cross-validation
    cv_folds: int = 5                     # fold della cross-validation temporale
    forecast_horizon_months: int = 1      # orizzonte di previsione
    view_z_threshold: float = 1.0         # soglia di selezione endogena delle view
    standardize_cross_section: bool = True
    clip_view_abs: float | None = None    # eventuale clipping delle view
    omega_window_months: int = 24         # finestra L per la varianza errori Omega

# -----------------------------------------------------------------------------
# Regime detection via Random Forest
# -----------------------------------------------------------------------------
@dataclass
class RegimeConfig:
    # --- Trigger che definiscono l'etichetta di stress (target del RF) -------
    realized_vol_window: int = 60         # 1° trigger: volatilità realizzata
    realized_vol_pct: float = 0.80
    drawdown_window_months: int = 12      # 2° trigger: drawdown del benchmark
    drawdown_threshold: float = -0.10
    vix_pct: float = 0.80                 # 3° trigger: livello VIX
    hy_oas_pct: float = 0.80              # 4° trigger: spread High Yield
    # --- Modalità e aggregazione dei trigger --------------------------------
    trigger_mode: str = "level"
    delta_horizon_months: int = 1         # orizzonte della variazione
    delta_pct: float = 0.90               # percentile espandente
    n_triggers_stress: int = 1            # stress se >= n trigger attivi
    # --- Random Forest -------------------------------------------------------
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 5
    class_weight: str | None = "balanced"
    min_train_months: int = 36
    feature_lag_months: int = 1           # feature laggate di 1 mese (point-in-time)
    # --- Etichetta e post-processing ----------------------------------------
    label_horizon_months: int = 1
    stress_prob_threshold: float = 0.5    # soglia sulla prob. lisciata per 'stress'
    prob_smoothing_window: int = 3        # finestra di smoothing (stabilizzatore)

# -----------------------------------------------------------------------------
# Vincoli di portafoglio (per regime)
# -----------------------------------------------------------------------------
@dataclass
class ConstraintConfig:
    long_only: bool = True                # pesi non negativi
    fully_invested: bool = True           # somma dei pesi = 1
    single_asset_cap: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.25, "stress": 0.15})
    # Deroga al single_asset_cap per oro
    asset_cap_override: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "normal": {},
        "stress": {"gold": 0.25},
    })
    # Vincoli di gruppo per leve di rischio, distinti per regime
    group_limits: Dict[str, Dict[str, Tuple[float | None, float | None]]] = field(
        default_factory=lambda: {
            "normal": {
                "defensive": (0.25, None),
                "equity":    (None, 0.60),
                "credit":    (None, 0.15),
                "gold":      (None, 0.15),
            },
            "stress": {
                "defensive": (0.50, None),
                "equity":    (None, 0.35),
                "credit":    (None, 0.10),
                "gold":      (None, 0.25),
            },
        })

# -----------------------------------------------------------------------------
# Controllo del rischio / target di volatilità
# -----------------------------------------------------------------------------
@dataclass
class RiskConfig:
    target_vol_annual: float = 0.10       # target di volatilità annua (non attivo)
    max_scaling: float = 1.0
    vol_estimation_window: int = 60

# -----------------------------------------------------------------------------
# Ottimizzatore media-varianza
# -----------------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    gamma: float = 4.5
    cov_lookback_days: int = 252
    cov_shrinkage: str = "ledoit_wolf"
    solver: str = "ECOS"

# -----------------------------------------------------------------------------
# Backtesting
# -----------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    # Struttura temporale del backtest rolling, senza look-ahead
    rebalance_freq: str = "ME"
    transaction_costs_bps: float = 10.0     # 5 bps/lato
    annualization_factor: int = MONTHS_PER_YEAR

# -----------------------------------------------------------------------------
# Feature engineering a livello di asset
# -----------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    reversal_months: int = 1
    momentum_specs: Tuple[Tuple[str, int, int], ...] = (
        ("Momentum_3_1", 1, 3),
        ("Momentum_12_1", 1, 12),
    )
    vol_windows: Tuple[int, ...] = (30, 60, 90)
    moment_window: int = 126
    drawdown_windows: Dict[str, int] = field(default_factory=lambda: {
        "6m": 126, "12m": 252})
    annualization_days: int = TRADING_DAYS_PER_YEAR
    require_full_window: bool = True

# -----------------------------------------------------------------------------
# Contenitore di configurazione globale
# -----------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = SEED
    data: DataConfig = field(default_factory=DataConfig)
    bl: BlackLittermanConfig = field(default_factory=BlackLittermanConfig)
    lasso: LassoConfig = field(default_factory=LassoConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    universe: Tuple[Asset, ...] = ASSET_UNIVERSE

    # ---- Accessori ----------------------------------------------------------
    @property
    def tickers(self) -> List[str]:
        return [a.ticker for a in self.universe]

    @property
    def asset_by_ticker(self) -> Dict[str, Asset]:
        return {a.ticker: a for a in self.universe}

    @property
    def currency_map(self) -> Dict[str, str]:
        return {a.ticker: a.quote_ccy for a in self.universe}

    def tickers_in_class(self, asset_class: str) -> List[str]:
        return [a.ticker for a in self.universe if a.asset_class == asset_class]

    def tickers_in_group(self, sub_group: str) -> List[str]:
        if sub_group == "defensive":
            return [a.ticker for a in self.universe if a.sub_group in ("cash", "govt")]
        if sub_group == "credit":
            return [a.ticker for a in self.universe if a.sub_group in ("credit_hy", "em_bond")]
        return [a.ticker for a in self.universe if a.sub_group == sub_group]

    def validate(self) -> None:
        w = self.bl.strategic_weights
        missing = set(self.tickers) - set(w)
        if missing:
            raise ValueError(f"strategic_weights mancanti per: {sorted(missing)}")
        total = sum(w.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"strategic_weights devono sommare a 1, somma={total:.6f}")
        if not 0 < self.risk.target_vol_annual < 1:
            raise ValueError("target_vol_annual fuori range (0,1).")
        if self.risk.max_scaling <= 0:
            raise ValueError("max_scaling deve essere > 0.")

# Istanza di default importabile dagli altri blocchi
CONFIG = Config()

if __name__ == "__main__":
    CONFIG.validate()
    print("Config valida.")
    print(f"  Asset ({len(CONFIG.tickers)}):", ", ".join(CONFIG.tickers))
    print("  Equity:", CONFIG.tickers_in_group("equity"))
    print("  Difensivi:", CONFIG.tickers_in_group("defensive"))
    print("  Somma w_mkt:", round(sum(CONFIG.bl.strategic_weights.values()), 6))
    print("  Valute:", {k: v for k, v in CONFIG.currency_map.items() if v != "EUR"})
    print("  data_dir:", CONFIG.data.data_dir)