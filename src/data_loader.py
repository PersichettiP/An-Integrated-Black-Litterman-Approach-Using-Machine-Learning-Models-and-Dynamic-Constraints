# =============================================================================
# DATALOADER.PY — Blocco 1 · Caricamento e pulizia dei dati
# =============================================================================
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
from config import Config, CONFIG

logger = logging.getLogger(__name__)

# Etichette possibili della colonna data nei vari export
_DATE_LABELS = ("Exchange Date", "DATE", "Date", "observation_date")
_MAX_HEADER_SCAN = 60

# -----------------------------------------------------------------------------
# Contenitore dei dati di mercato
# -----------------------------------------------------------------------------
@dataclass
class MarketData:
    prices_local: pd.DataFrame          # prezzi nella valuta di quotazione
    prices_eur: pd.DataFrame            # prezzi convertiti in EUR
    returns: pd.DataFrame               # rendimenti log giornalieri
    fx: pd.DataFrame                    # cambi utilizzati
    macro: Dict[str, pd.DataFrame]      # tabelle macro-finanziarie
    benchmark: pd.Series                # portafoglio di mercato
    availability: pd.DataFrame          # copertura temporale per asset
    fx_applied: bool = False

# -----------------------------------------------------------------------------
# Parsing di basso livello degli export Excel
# -----------------------------------------------------------------------------
def _make_unique(columns: Iterable) -> List[str]:
    # Rende univoci i nomi di colonna (gestisce duplicati e celle vuote)
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in columns:
        name = "Unnamed" if (c is None or (isinstance(c, float) and pd.isna(c))) else str(c).strip()
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out

def _locate_header(raw: pd.DataFrame, date_labels: Iterable[str] = _DATE_LABELS) -> int:
    # Individua la riga di intestazione cercando una colonna data nota
    labels = {str(x).strip().lower() for x in date_labels}
    limit = min(_MAX_HEADER_SCAN, len(raw))
    for i in range(limit):
        row_vals = {str(v).strip().lower() for v in raw.iloc[i].tolist() if pd.notna(v)}
        if row_vals & labels:
            return i
    raise ValueError("Intestazione con colonna data non trovata.")

def _read_table(path: Path) -> pd.DataFrame:
    # Legge un export Excel e restituisce la tabella indicizzata per data
    raw = pd.read_excel(path, header=None)
    hdr = _locate_header(raw)
    columns = _make_unique(raw.iloc[hdr].tolist())
    table = raw.iloc[hdr + 1:].copy()
    table.columns = columns
    date_col = next((c for c in columns if c in _DATE_LABELS), None)
    if date_col is None:
        date_col = columns[0]
    table[date_col] = pd.to_datetime(table[date_col], errors="coerce")
    table = table.dropna(subset=[date_col])
    table = table.set_index(date_col).sort_index()
    table.index.name = "date"
    for c in table.columns:
        table[c] = pd.to_numeric(table[c], errors="coerce")
    table = table[~table.index.duplicated(keep="last")]
    return table

def _value_column(table: pd.DataFrame, filename: str = "") -> pd.Series:
    # Estrae la serie-valore: 'Close' se presente, altrimenti la 1a colonna numerica (RI)
    if "Close" in table.columns:
        return table["Close"].rename("value")
    numeric = table.select_dtypes("number").dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError(f"{filename}: nessuna colonna numerica di valore trovata.")
    return numeric.iloc[:, 0].rename("value")

# -----------------------------------------------------------------------------
# Data loader: orchestrazione del caricamento
# -----------------------------------------------------------------------------
class DataLoader:
    # Carica e armonizza tutti i dati del framework a partire dalla configurazione
    def __init__(self, config: Config = CONFIG) -> None:
        self.cfg = config
        self.data_dir = Path(config.data.data_dir)

    # ---- Prezzi ETF ---------------------------------------------------------
    def _load_price(self, filename: str) -> pd.Series:
        # Serie total-return (RI) giornaliera di un singolo ETF
        path = self.data_dir / filename
        table = _read_table(path)
        return _value_column(table, filename)

    def load_prices_local(self) -> pd.DataFrame:
        series: Dict[str, pd.Series] = {}
        for asset in self.cfg.universe:
            try:
                series[asset.ticker] = self._load_price(asset.filename)
                logger.info("Caricato %s da %s (%d oss.)",
                            asset.ticker, asset.filename, series[asset.ticker].notna().sum())
            except FileNotFoundError:
                logger.error("File mancante per %s: %s (cercato in %s)",
                             asset.ticker, asset.filename, self.data_dir)
                raise
        return pd.DataFrame(series)[self.cfg.tickers]

    # ---- Cambi (fonte FRED) -------------------------------------------------
    def _read_fred(self, series_id: str) -> Optional[pd.Series]:
        for ext in (self.cfg.data.fred_file_ext, "csv", "xlsx"):
            path = Path(self.cfg.data.fred_dir) / f"{series_id}.{ext}"
            if path.exists():
                break
        else:
            logger.warning("Serie FRED mancante: %s (.csv/.xlsx) in %s",
                           series_id, self.cfg.data.fred_dir)
            return None
        raw = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        date_col = next((c for c in raw.columns if str(c).strip() in _DATE_LABELS), raw.columns[0])
        val_col = next((c for c in raw.columns if c != date_col), None)
        s = pd.Series(
            pd.to_numeric(raw[val_col], errors="coerce").values,
            index=pd.to_datetime(raw[date_col], errors="coerce"),
            name=series_id,
        ).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        logger.info("Caricata serie FRED %s (%d oss., %s -> %s).",
                    series_id, len(s), s.index.min().date(), s.index.max().date())
        return s

    def load_fx(self) -> pd.DataFrame:
        # Costruisce i cambi necessari da serie FRED
        base = self.cfg.data.base_ccy
        needed = {a.quote_ccy for a in self.cfg.universe if a.quote_ccy != base}
        if self.cfg.data.benchmark_ccy != base:
            needed.add(self.cfg.data.benchmark_ccy)
        ids = {sid for ccy, sid in self.cfg.data.fx_direct.items() if ccy in needed}
        fred = {sid: self._read_fred(sid) for sid in sorted(ids)}
        fred = {k: v for k, v in fred.items() if v is not None}
        fx_series: Dict[str, pd.Series] = {}
        for ccy, sid in self.cfg.data.fx_direct.items():
            if ccy in needed and sid in fred:
                fx_series[ccy] = fred[sid].rename(ccy)
        return pd.DataFrame(fx_series) if fx_series else pd.DataFrame()

    def convert_to_eur(self, prices_local: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        # Converte i prezzi in EUR dividendo per il cambio "valuta/EUR" allineato
        base = self.cfg.data.base_ccy
        prices_eur = prices_local.copy()
        fx_aligned = (fx.reindex(prices_local.index).ffill(limit=self.cfg.data.max_ffill_days)
                      if not fx.empty else fx)
        all_converted = True
        for asset in self.cfg.universe:
            ccy = asset.quote_ccy
            if ccy == base:
                continue
            if (not fx.empty) and (ccy in fx_aligned.columns):
                prices_eur[asset.ticker] = prices_local[asset.ticker] / fx_aligned[ccy]
            else:
                all_converted = False
                logger.warning("%s resta in %s (cambio non disponibile).", asset.ticker, ccy)
        return prices_eur, all_converted

    # ---- Macro e benchmark --------------------------------------------------
    def load_macro(self) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for name, fname in self.cfg.data.macro_files.items():
            path = self.data_dir / fname
            if not path.exists():
                logger.warning("File macro mancante: %s (%s).", fname, name)
                continue
            out[name] = _read_table(path)
            logger.info("Caricata macro %s da %s (%d righe).", name, fname, len(out[name]))
        return out

    def load_benchmark(self, fx: pd.DataFrame) -> pd.Series:
        # Serie del benchmark, convertita in EUR se necessario
        path = self.data_dir / self.cfg.data.benchmark_file
        table = _read_table(path)
        bench = _value_column(table, self.cfg.data.benchmark_file).rename(
            self.cfg.data.benchmark_ticker)
        bccy = self.cfg.data.benchmark_ccy
        if bccy != self.cfg.data.base_ccy and (not fx.empty) and (bccy in fx.columns):
            fx_aligned = fx[bccy].reindex(bench.index).ffill(limit=self.cfg.data.max_ffill_days)
            bench = bench / fx_aligned
            logger.info("Benchmark convertito in %s.", self.cfg.data.base_ccy)
        elif bccy != self.cfg.data.base_ccy:
            logger.warning("Benchmark lasciato in %s (cambio non disponibile).", bccy)
        return bench

    # ---- Orchestrazione -----------------------------------------------------
    def load(self) -> MarketData:
        # Pipeline completa: prezzi -> FX -> EUR -> rendimenti -> macro/benchmark
        self.cfg.validate()
        prices_local = self.load_prices_local()
        fx = self.load_fx()
        prices_eur, fx_applied = self.convert_to_eur(prices_local, fx)
        start, end = self.cfg.data.start_date, self.cfg.data.end_date
        prices_local = prices_local.loc[start:end]
        prices_eur = prices_eur.loc[start:end]
        limit = self.cfg.data.max_ffill_days
        prices_eur_f = prices_eur.ffill(limit=limit)
        returns = np.log(prices_eur_f / prices_eur_f.shift(1)).replace([np.inf, -np.inf], np.nan)
        macro = self.load_macro()
        benchmark = self.load_benchmark(fx).loc[start:end]
        availability = pd.DataFrame({
            "first": prices_eur.apply(lambda s: s.first_valid_index()),
            "last": prices_eur.apply(lambda s: s.last_valid_index()),
            "n_obs": prices_eur.notna().sum(),
        })
        if not fx_applied:
            logger.warning("CONVERSIONE FX INCOMPLETA: fornire i file cambio mancanti.")
        return MarketData(prices_local=prices_local, prices_eur=prices_eur, returns=returns,
                          fx=fx, macro=macro, benchmark=benchmark,
                          availability=availability, fx_applied=fx_applied)

    # ---- Utility ------------------------------------------------------------
    @staticmethod
    def to_month_end(df, how: str = "last"):
        # Ricampiona a fine mese ("last" per i livelli, "sum" per i flussi).
        rule = "ME"
        if how == "last":
            return df.resample(rule).last()
        if how == "sum":
            return df.resample(rule).sum(min_count=1)
        raise ValueError("how deve essere 'last' o 'sum'.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    md = DataLoader(CONFIG).load()
    print("\n=== PREZZI (EUR) ===")
    print("shape:", md.prices_eur.shape, "| range:",
          md.prices_eur.index.min().date(), "->", md.prices_eur.index.max().date())
    print("\n=== DISPONIBILITA PER ASSET ===")
    print(md.availability.to_string())
    print("\nFX applicata pienamente:", md.fx_applied, "| cross:", list(md.fx.columns))