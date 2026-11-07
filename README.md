# Engulfing + SMA50 backtest (H1)

> ⚠️ Sur ~2,8 ans de données H1, la stratégie était **perdante** (PF 0.91, −130 R sur 2108 trades).
> Ce dépôt est un outil de recherche, pas un système à trader.

## Règles
| Règle | Valeur |
|---|---|
| Unité de temps | H1 |
| Actifs | EURUSD, GBPUSD, USDJPY |
| Achat | bougie haussière dont le corps englobe le corps baissier précédent, clôture > SMA50 |
| Vente | symétrique, clôture < SMA50 |
| Entrée | ouverture de la bougie suivante (+ coût) |
| Stop Loss | 1.5 × ATR(14) |
| Take Profit | 2 × risque (R:R 1:2) |
| Gestion | 1 position à la fois, 1 % de risque, SL prioritaire si SL et TP dans la même bougie, sortie à l'open en cas de gap |

Tous les réglages sont en tête de `strategy.py`.

## Structure
| Fichier | Rôle |
|---|---|
| `strategy.py` | règles + moteur : `load_csv`, `compute_atr`, `detect_engulfing`, `signals`, `backtest`, `stats` |
| `run.py` | backtest de tous les actifs, résumé, export `data/trades.csv` |
| `fetch_dukascopy_h1.py` | télécharge 10 ans de H1 depuis Dukascopy (via `curl`) → `data/<PAIR>_H1.csv` |
| `test_strategy.py` | tests unitaires (13) |
| `_archive/` | anciennes versions (SMC, sweep, optimiseur Optuna…) |

`run.py` lit `data/<PAIR>_H1.csv` (source unique : Dukascopy). Horodatages en UTC (Paris = UTC+1 l'hiver, UTC+2 l'été).
Format CSV attendu : `timestamp, open, high, low, close[, volume]` (UTC).

## Commandes
```
.venv/bin/python fetch_dukascopy_h1.py --years 10    # données Dukascopy (long)
.venv/bin/python run.py                              # backtest
.venv/bin/python -m pytest -q                        # tests
```
