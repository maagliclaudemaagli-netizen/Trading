# Extraction Polygon pour essais uniquement

Cet exemple isolé ne modifie pas le moteur ni sa source de données OANDA.
La clé est lue depuis `POLYGON_API_KEY`, jamais stockée dans le code.

Depuis la racine du dépôt :

```bash
.venv/bin/python -m venv .venv/polygon-tests
.venv/polygon-tests/bin/python -m pip install -r tests/requirements-polygon.txt
read -r -s POLYGON_API_KEY
export POLYGON_API_KEY
.venv/polygon-tests/bin/python tests/polygon_extract.py \
  --ticker C:EURUSD --timespan day \
  --from-date 2025-01-01 --to-date 2025-12-31 \
  --output data/polygon_EURUSD_day.csv
unset POLYGON_API_KEY
```

Saisir la clé après la commande `read`. Sans `--output`, seul un aperçu est
affiché. Les horodatages sont en UTC ; le CSV contient
`Timestamp,Open,High,Low,Close,Volume`. Il n'est pas destiné à être importé
directement dans le backtest M5 OANDA.

L'environnement dédié évite les conflits entre les dépendances de Polygon
et celles du projet. Le multiplicateur reste fixé à 1 dans cet exemple.

Les fenêtres de 30 jours évitent la troncature à 50 000 bougies de base
(43 200 minutes maximum). Chaque appel est espacé d'au moins 12,1 secondes,
y compris entre deux extractions partageant le même limiteur. Les nouvelles
tentatives automatiques du SDK sont désactivées : une erreur HTTP, notamment
429, est propagée sans annoncer un succès ni exporter des données partielles.
Ne pas lancer plusieurs instances ou utiliser la même clé ailleurs en
parallèle : la limite s'applique à tous les appels du compte. L'accès à
l'historique Forex dépend aussi des droits de l'abonnement.

Tests hors ligne, sans clé ni appel réseau :

```bash
.venv/polygon-tests/bin/python -B -m unittest discover -s tests -p 'test_polygon_extract.py' -v
```

## Fichiers Massive S3

Le script [massive_s3_extract.py](massive_s3_extract.py) utilise les clés
**S3**, distinctes de la clé REST Polygon. Le point d'accès est
`https://files.massive.com`, le bucket `flatfiles`, et les requêtes sont
signées en AWS Signature V4. Aucun identifiant n'est enregistré dans le code.
Révoquer toute clé partagée publiquement ou dans un chat.

```bash
.venv/bin/python -m venv .venv/s3-tests
.venv/s3-tests/bin/python -m pip install -r tests/requirements-s3.txt
read -r -s AWS_ACCESS_KEY_ID
export AWS_ACCESS_KEY_ID
read -r -s AWS_SECRET_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY
.venv/s3-tests/bin/python tests/massive_s3_extract.py \
  --date 2025-01-02 --timespan day --ticker C:EURUSD \
  --output data/massive_EURUSD_day_2025-01-02.csv
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
```

Saisir chaque identifiant après son `read`. Les fichiers sont organisés sous
`global_forex/{minute|day}_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`.
Un fichier contient plusieurs paires : la lecture est effectuée par blocs,
puis filtrée sur la paire demandée. Les horodatages source en nanosecondes
sont convertis en UTC. Le CSV exporté contient
`timestamp,open,high,low,close,volume`.

Les erreurs d'accès S3, fichiers absents, gzip corrompus, schémas incomplets
et bougies incohérentes sont propagées, sans exporter de résultat partiel.
La disponibilité des fichiers dépend des droits de l'abonnement. Ces
données de test ne remplacent pas automatiquement les bougies M5 OANDA ;
`minute` désigne M1 et `day` D1, pas M5. Leur origine ne garantit ni leur
exhaustivité ni la rentabilité d'un backtest.

Tests hors ligne :

```bash
.venv/s3-tests/bin/python -B -m unittest discover -s tests -p 'test_massive_s3_extract.py' -v
```

## Comparaison des données et backtest de la stratégie

[massive_s3_backtest.py](massive_s3_backtest.py) compare **EURUSD** entre
les fichiers S3 M1 et le cache OANDA M5 local, puis exécute le moteur
institutionnel existant sur chaque source, sans modifier ses règles.
Installer à nouveau les dépendances S3 pour disposer aussi de `pyarrow`.

Après avoir configuré les deux variables S3 comme ci-dessus :

```bash
.venv/s3-tests/bin/python -m pip install -r tests/requirements-s3.txt
.venv/s3-tests/bin/python tests/massive_s3_backtest.py \
  --start 2025-01-01 --end 2025-02-01 \
  --cache data/cache/oanda/EURUSD_M5 \
  --alignment intraday --price-basis mid \
  --output reports/s3_validation_eurusd_2025_01.json
```

La date de fin est exclusive. Le catalogue S3 est parcouru avec pagination :
seuls les fichiers minute effectivement listés pour la période sont lus.
Un fichier listé mais refusé en lecture (HTTP 403) arrête le test, sans
produire de rapport Massive ni réutiliser des données OANDA à sa place.

Chaque M5 doit contenir **exactement cinq M1**, triées, uniques et alignées
en UTC. Open = première minute, high = maximum, low = minimum,
close = dernière minute, volume = somme. Toute M5 partielle provoque un
échec explicite ; aucune minute ni bougie n'est remplie ou inventée.
Les intervalles complètement vides et les dates sans fichier sont signalés,
sans conclure automatiquement qu'il s'agit de jours de fermeture.

Le JSON mesure la couverture commune, les timestamps propres à chaque
source, les écarts OHLC absolus moyens/maximaux et les empreintes SHA-256.
Les volumes ne sont pas directement comparés entre fournisseurs. Aucun
seuil d'acceptation de prix n'est supposé : ces mesures ne certifient pas
l'exhaustivité ou l'exactitude d'un fournisseur.

Le même alignement, spread fixe et choix `mid`/`bid` sont appliqués aux deux
backtests ; la nature réelle des cotations Massive reste à confirmer auprès
du fournisseur. Les tests démarrent au début de la période, sans historique
de préchauffage. Chaque source conserve ses propres bougies : des différences
de couverture ou de cotations peuvent changer les signaux et les trades.
La rentabilité du backtest n'est pas une validation des données.

Les exports associés sont `.s3.m5.csv`, `.oanda.m5.csv`,
`.s3.trades.csv`, `.oanda.trades.csv`, `.s3.audit.csv`, `.oanda.audit.csv`.

Tests synthétiques hors ligne, sans clés ni appels réseau :

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_massive_s3*.py' -v
```
