# Framework di asset allocation multi-asset con LASSO, Black–Litterman e regime detection

Implementazione Python del framework sviluppato nella tesi di laurea magistrale
*"[titolo della tesi]"*, LUISS Guido Carli — A.A. 2025/2026.

Il framework integra:
- **views generate tramite LASSO** su un pannello di 27 predittori (feature specifiche
  di asset e variabili macro-finanziarie);
- **Black–Litterman** per la sintesi bayesiana tra rendimenti di equilibrio e views,
  validato contro [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt);
- **identificazione dei regimi di mercato** tramite Random Forest, con vincoli
  allocativi dinamici in funzione del regime (normale/stress).

Il backtest copre l'universo di 12 ETF UCITS descritto nel Capitolo 3 della tesi,
su un orizzonte di 156 mesi (gennaio 2013 – dicembre 2025), fuori campione,
con finestra di stima espansiva e poi mobile.

## Struttura del repository

```
.
├── src/                        # moduli del framework
│   ├── config.py                # configurazione centralizzata (universo, parametri, vincoli)
│   ├── data_loader.py            # caricamento e allineamento dei dati di prezzo
│   ├── feature_engineering.py    # costruzione delle feature per il modulo LASSO
│   ├── lasso_views.py             # generazione delle views tramite LASSO
│   ├── black_litterman.py         # modello Black–Litterman
│   ├── regime_detection.py        # classificazione del regime di mercato (Random Forest)
│   ├── constraints.py             # vincoli allocativi dinamici regime-based
│   ├── optimizer.py               # ottimizzazione del portafoglio
│   ├── backtest.py                # motore di backtest e generazione delle varianti
│   ├── diagnostics.py             # diagnostica statistica e tabelle di risultato
│   └── plots.py                   # grafici e tema visivo condiviso
├── notebooks/
│   └── Notebook.ipynb             # esecuzione completa del backtest e dei risultati del Capitolo 4
├── requirements.txt
└── .gitignore
```

## Dati

I dati di input (export Excel per i 12 ETF dell'universo investibile e le serie
macro-finanziarie) non sono inclusi nel repository per motivi di dimensione e di
licenza sulle fonti. Gli ISIN e i provider degli strumenti sono elencati nel
Capitolo 3 (§3.8) della tesi; chi desidera replicare il backtest deve procurarsi
autonomamente le serie storiche e posizionarle in una cartella `data/` nella
directory principale, con lo stesso schema di colonne atteso da `data_loader.py`.

## Setup

```bash
python -m venv venv
source venv/bin/activate    # su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Utilizzo

Il notebook in `notebooks/Notebook.ipynb` contiene l'intera pipeline — dal
caricamento dei dati alla generazione delle tabelle e dei grafici riportati nel
Capitolo 4 della tesi — ed è il punto di ingresso più diretto per esplorare i
risultati. Va eseguito con la cartella `src/` nel path di Python (ad esempio
avviando Jupyter dalla directory principale del repository).

## Riproducibilità

Seed fissato a 42 in `config.py` per la riproducibilità dei test bootstrap e
delle componenti stocastiche (Random Forest, LASSO con cross-validation).

## Riferimento

Persichetti, P. (2026). *[Titolo della tesi]*. Tesi di laurea magistrale,
LUISS Guido Carli. Relatore: Prof. Ugo Pomante.
