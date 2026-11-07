# Backtest de la stratégie institutionnelle SMC / ICT

Ce dépôt contient un moteur Python de recherche et de backtest pour la
stratégie institutionnelle décrite dans [INSTITUTIONAL.md](INSTITUTIONAL.md).
Il ne constitue pas un système de trading prêt à l'emploi.

## Principe

La combinaison intraday utilisée par défaut associe la liquidité Daily, la
confirmation H1 et la clôture de validation H4. Le moteur recherche, dans
l'ordre, un contact avec une zone interne, une cassure de structure, au moins
deux zones opposées invalidées et deux zones créées dans le sens du mouvement.
La validation attend une clôture directionnelle au-delà du niveau clé.

L'entrée attend un repli sur une zone créée ; le stop est structurel et la
cible correspond à une liquidité externe. Les règles sont symétriques pour
les achats et les ventes. Les définitions opérationnelles, les limites des
données OHLC et les deux autres combinaisons fractales sont détaillées dans
[INSTITUTIONAL.md](INSTITUTIONAL.md).

## Organisation

- [institutional/](institutional/) : données, zones, stratégie et backtest.
- [INSTITUTIONAL.md](INSTITUTIONAL.md) : conventions et protocole exécuté.
- [requirements.txt](requirements.txt) : dépendances Python.
- [tests/](tests/) : tests de régression sur données synthétiques.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Utilisation

Fournir un fichier local de bougies M5 OANDA, horodatées en UTC, avec les
colonnes `timestamp,open,high,low,close,volume` ; `time` en secondes Unix peut
remplacer `timestamp`. Aucun téléchargement n'est effectué automatiquement.
Les données de marché ne sont pas incluses dans le répertoire du projet.

```bash
.venv/bin/python -m institutional \
  --csv chemin/vers/EURUSD_M5.csv \
  --pair EURUSD \
  --output reports/institutional.json
```

Le rapport JSON est accompagné des fichiers `institutional.trades.csv` et
`institutional.audit.csv`. Le dossier de sortie est créé lors de l'exécution.
Les données locales et les rapports générés ne sont pas versionnés.

## Vérifications

```bash
.venv/bin/python -B -m institutional --help
.venv/bin/python -B -m unittest discover -s tests -v
```

Les tests utilisent uniquement des données synthétiques et ne mesurent pas
la rentabilité sur le marché.

Les résultats d'un backtest dépendent des données et des conventions de
simulation et ne garantissent pas la rentabilité future.
