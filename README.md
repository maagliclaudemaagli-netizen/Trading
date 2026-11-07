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

Fournir un export local de bougies M5 Dukascopy, horodatées en UTC, avec les
colonnes `timestamp,open,high,low,close,volume` ; `time` en secondes Unix peut
remplacer `timestamp`. Aucun téléchargement n'est effectué automatiquement.
Les données de marché ne sont pas incluses dans le dépôt versionné.

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

Les tests exécutés par défaut utilisent des données synthétiques et ne
mesurent pas la rentabilité sur le marché.

Les résultats d'un backtest dépendent des données et des conventions de
simulation et ne garantissent pas la rentabilité future.

## Moteur de données historiques Dukascopy

[forex_historical_engine/](forex_historical_engine/) fournit la seule source
de données du projet : téléchargement officiel Dukascopy,
archives brutes conservées, Parquet, suivi SQLite, reconstruction OHLC,
contrôles qualité, statistiques, nettoyage explicite et dashboard Streamlit.
Le backtest utilise les exports CSV produits par ce moteur.

### Prérequis

Python 3.11 ou supérieur compatible avec les dépendances est requis.
L'environnement de validation utilise Python 3.15, Pandas 3.1.1,
NumPy 2.5.4, PyArrow 26.0.1, Requests 2.34.2, Streamlit 1.67.0 et Pytest 9.2.0.
Aucune clé API n'est nécessaire pour les archives utilisées.

### Ticks bruts avec plafond de stockage

Pour demander les 23 paires configurées et essayer également XAU/USD sur
2025, dans un dossier dédié limité à 1 Gio :

```bash
.venv/bin/python -B main.py download-ticks \
  --pairs all --include-xauusd \
  --start 2025-01-01 --end 2025-12-31 \
  --max-bytes 1073741824 --output-dir data/dukascopy_ticks_2025
```

Le parcours est séquentiel, heure par heure et instrument par instrument.
La structure LZMA et la longueur des enregistrements `.bi5` sont vérifiées
avant leur conservation sous `raw/` ; les archives déjà présentes sont
revérifiées et réutilisées. Cela ne certifie pas la validité des cours. Les
fichiers déjà présents dans ce dossier comptent dans le plafond et ne sont
pas supprimés.
Le téléchargement s'arrête avant qu'une archive ne fasse dépasser le
plafond, y compris pendant la publication du fichier temporaire. Une marge
de la taille de l'archive est réservée pour cette publication sans écrasement ;
le stockage final peut donc rester inférieur au plafond.
Le résumé indique les archives absentes et, en cas d'arrêt sur
limite, la prochaine archive ; le code de sortie est `2` en cas de limite
ou d'archives absentes, pas un succès complet. Les archives absentes ne
prouvent pas une fermeture de marché.

Le plafond porte sur la somme des tailles des fichiers du dossier dédié.
Un verrou empêche deux téléchargements simultanés ; ne pas modifier ce
dossier par un autre programme en parallèle. Si une interruption brutale
laisse `.download-ticks.lock`, vérifier qu'aucun téléchargement n'est actif
avant de retirer manuellement ce dossier de verrou vide.
Cette commande ne produit ni Parquet, ni cache, ni rapport. Un traitement ou
une reconstruction ultérieure crée des fichiers supplémentaires et n'est pas
couvert par ce plafond ; prévoir l'espace avant de les lancer.
La disponibilité de XAU/USD et son diviseur de prix restent à vérifier :
la récupération brute ne suppose pas d'échelle de prix.

Un accès réseau autorisé à `datafeed.dukascopy.com` est nécessaire. Le
plafond, les droits d'accès et les archives disponibles peuvent empêcher
de couvrir toute la période demandée.

### Installation et premier téléchargement

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py download --pair EURUSD --timeframe M1 \
  --start 2025-01-06 --end 2025-01-06
.venv/bin/python main.py download --pair EURUSD --timeframe tick \
  --start 2025-01-06 --end 2025-01-06
.venv/bin/python main.py download --pairs majors --timeframe M5 --years 1
.venv/bin/python main.py download --pair EURUSD --timeframe H1 --years 5
```

Les dates sont des jours UTC entiers, début et fin **inclus**. Sans dates,
les sept derniers jours terminés sont demandés. `--years` accepte
1, 5, 10, 15 et 20. Le jour courant et les dates futures sont refusés :
un HTTP 200 sur une archive en cours ne prouve pas sa complétude.
Les groupes `majors`, `minors` et `all` couvrent les 23 paires du cahier
des charges. La disponibilité réelle est vérifiée à chaque demande.
Pour ajouter un instrument, fournir son diviseur de prix **vérifié** avec
`--price-scale`. Une échelle inconnue n'est jamais déduite automatiquement.

```bash
.venv/bin/python main.py availability --pair EURUSD \
  --start 2005-01-01 --end 2005-01-10
.venv/bin/python main.py download --pair EURUSD --timeframe M1 \
  --since-first --search-start 2005-01-01 --end 2025-01-06
```

`availability` cherche séparément les premières archives M1 Bid et Ask
dans l'intervalle demandé et conserve les fichiers vérifiés. Cette recherche
peut être longue. `--since-first` signifie **depuis la première archive trouvée
dans cet intervalle**, et non depuis un début historique absolu certifié.
Les ticks ont une couverture indépendante. L'API officielle de métadonnées
a répondu HTTP 429 (« Bot blocked ») lors de l'étude ; elle n'est pas
contournée. La documentation web n'était pas accessible dans l'environnement.

### Source, formats et provenance

- [Archives officielles](https://datafeed.dukascopy.com/datafeed/) :
  `{instrument}/{année}/{mois zéro-indexé}/{jour}/`.
- Ticks : `12h_ticks.bi5`, LZMA, enregistrements big-endian
  `>IIIff` : millisecondes dans l'heure, Ask entier, Bid entier,
  volume Ask, volume Bid.
- M1 originaux : `BID_candles_min_1.bi5` et
  `ASK_candles_min_1.bi5`, LZMA, `>IIIIIf` : secondes dans le jour,
  Open, Close, Low, High entiers, volume source.
- [Page historique Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
  et [documentation export](https://wiki.dukascopy.com/en/development/data-export/)
  constituent les références officielles à consulter pour les conditions d'accès.

Les deux formats binaires ont été vérifiés sur les archives EUR/USD du
6 janvier 2025. Un mois dans l'URL vaut `mois calendrier - 1`.
Les données représentent les cotations de Dukascopy, pas les cotations
universelles de tous les brokers Forex Spot.

Les colonnes communes comprennent `timestamp` UTC, `instrument`, `source`,
`source_url` et `source_record`. Les données originales conservent également
`timestamp_original`, `origin_utc`, les prix entiers `*_raw` et `price_scale`.
Les ticks ont `bid`, `ask`, `bid_volume`, `ask_volume` ; les bougies ont
`side`, `open`, `high`, `low`, `close`, `volume`, `timeframe`,
`reconstructed` et `volume_kind`. Les tailles/volumes source sont
**indicatifs et propres au fournisseur**, pas un volume exécuté global.
Leur unité économique n'est pas certifiée par ce projet.

### Stockage, reprise et erreurs

```text
data/raw/EURUSD/2025/00/06/BID_candles_min_1.bi5
data/processed/EURUSD/M1/2025/01.parquet
data/processed/EURUSD/tick/2025/01/2025-01-06-12.parquet
data/cache/EURUSD/M1/2025/01/2025-01-06-bid.parquet
data/reports/EURUSD/M1.json
data/metadata.sqlite
data/logs/download.log
```

Les ticks sont partitionnés par mois **et par heure** pour ne pas charger
plusieurs années en mémoire. Les bougies sont partitionnées mensuellement.
Compression Zstandard, écritures Parquet et archives via fichiers
temporaires puis remplacement atomique, contrôle LZMA et SHA-256.
SQLite conserve les statuts, erreurs, nombres de lignes, empreintes des
archives et partitions, ainsi que les rapports qualité.
Les partitions sont triées de façon stable par timestamp/côté pour
l'exploitation ; l'ordre original reste identifiable par `source_record`.
Un ordre source anormal est journalisé avant ce tri, sans modifier le brut.
Un fichier brut corrompu/modifié est signalé, jamais écrasé silencieusement.

Relancer exactement la même commande reprend les périodes absentes,
réutilise les archives et partitions vérifiées et régénère les sorties
manquantes. Les événements tick simultanés sont conservés. Le remplacement
d'une archive déjà traitée se fait par son identité, pas par son timestamp.
Le téléchargement incrémental peut aussi utiliser un nouveau `--start`.
Ne pas lancer plusieurs processus écrivains sur le même dossier de données.

Chaque paquet récupéré est aussi enregistré immédiatement en **Parquet**
dans le cache, avant la fusion des partitions mensuelles. Ce cache est relu
en priorité : même si le brut ou une sortie traitée n'existe plus, aucun
téléchargement n'est nécessaire pour ce paquet. `--cache` choisit son dossier,
avec `data/cache` par défaut (ou `{data-dir}/cache`). Les contrôles d'intégrité,
de provenance et d'échelle de prix restent appliqués aux données en cache.

Les options globales se placent **avant** la sous-commande :

```bash
.venv/bin/python main.py --data-dir data --workers 2 --delay 1.0 \
  --retries 3 --timeout 30 download --pair EURUSD --timeframe M1 \
  --start 2025-01-06 --end 2025-01-06
```

Les erreurs réseau/HTTP 5xx sont journalisées et réessayées avec délai.
Un HTTP **429 ordinaire** déclenche des attentes de **5, 15 puis 30 secondes**.
Il s'agit de **trois réessais maximum après la requête initiale**, soit au
plus quatre requêtes par paquet avec `--retries 3`. Un `Retry-After` plus
long est respecté. Une pause systématique de **1 seconde minimum** précède
chaque requête HTTP, y compris les réessais ; `--delay` ne peut pas être
inférieur à 1. Les requêtes et leurs délais de refroidissement partagent une
porte d'accès unique entre les workers, sans requêtes concurrentes pendant
un backoff.
Une fois les réessais épuisés, le paquet reçoit un statut `failed`, un
avertissement est journalisé et le téléchargement poursuit le paquet
suivant. Le résultat reste explicitement incomplet, avec `failed_archives`.
HTTP 401/403 et un HTTP 429 portant un refus explicite **« Bot blocked »**
interrompent toujours l'accès ; aucune restriction n'est contournée.
Une archive vide/404/204 reste indisponible, pas réussie.
Codes CLI : `0` commande terminée, `1` erreur, `2` demande d'archives ou
recherche de disponibilité incomplète. Une fermeture de marché peut donc
produire `2` sans prouver une panne. Un statut `verified` certifie le fichier,
**pas la qualité ni l'exhaustivité des cotations**.

L'option `--cache` est également acceptée après `download` ou `availability` :

```bash
.venv/bin/python main.py download --pair EURUSD --timeframe M1 \
  --start 2025-01-06 --end 2025-01-06 --cache data/dukascopy_cache
```

### Bougies, qualité et analyses

M1 peut être téléchargé directement. M5, M15, M30, H1, H4, D1, W1 et MN1
sont reconstruits depuis les M1 originaux, séparément pour Bid et Ask.
Les intervalles sont `[ouverture, clôture)` en UTC ; jours à 00:00,
semaines lundi 00:00, mois le premier à 00:00. Les semaines traversant les
mois/années sont agrégées avec les M1 du mois précédent. Open/Close prennent
le premier/dernier événement observé, High/Low le maximum/minimum.
Aucune bougie vide n'est créée. Les bougies partielles restent explicitement
`completeness=not_certified` ; `source_count`, `closed_at` et les premiers/
derniers timestamps source permettent leur inspection.
`input_sources` conserve en JSON les URL des archives brutes ayant contribué
à chaque bougie, y compris après plusieurs niveaux de reconstruction.

```bash
.venv/bin/python main.py rebuild --pair EURUSD --timeframe H4
.venv/bin/python main.py rebuild --pair EURUSD --timeframe M5 --from-ticks
.venv/bin/python main.py validate --pair EURUSD --timeframe M1
.venv/bin/python main.py report --pair EURUSD --timeframe M5
.venv/bin/python main.py status
```

Les reconstructions depuis les ticks sont séparées sous
`M1_from_ticks`, `M5_from_ticks`, etc. Mid est calculé **par tick**
`(bid + ask) / 2`, puis agrégé : jamais une moyenne d'OHLC Bid/Ask.
Les volumes des bougies issues des ticks restent absents (`NaN`), sans
inventer de volume exécuté.
Ces exports ne sont pas directement compatibles avec le backtest OHLCV,
qui exige un volume numérique : le parcours de backtest actuel utilise les
M5 reconstruites depuis les M1 d'origine, pas les archives de ticks brutes.

Le contrôle détecte prix non finis/nuls/négatifs, incohérences OHLC,
spreads négatifs, candidats Bid/Ask croisés, timestamps désordonnés,
doublons et sauts de prix supérieurs à 1 %. Les ticks partageant un timestamp
ne sont pas automatiquement des doublons.
Les trous de cotations sont des **candidats** : calendrier nominal Forex
dimanche 17:00 à vendredi 17:00 `America/New_York`, avec DST via `zoneinfo`.
Il n'existe pas de calendrier universel de jours fériés Forex dans le moteur.
Les fermetures UTC vérifiées peuvent être exclues avec `--holiday YYYY-MM-DD`.
Les interruptions propres à Dukascopy nécessitent une revue.

Les rapports incluent score de validité des lignes (formule documentée,
distincte de la complétude), couverture nominale, archives absentes,
périodes manquantes, nombres de ticks/bougies/jours, dates disponibles,
spreads moyen/médian/minimum/maximum, moyennes par heure et sessions UTC,
ainsi que leurs distributions (médiane, minimum, maximum et écart-type),
écart-type non annualisé des rendements logarithmiques Bid et amplitude
moyenne. Le spread des bougies est celui des **Close synchronisés**, pas un
spread tick intrabougie. La médiane est exacte, calculée dans une base SQLite
temporaire sur disque pour limiter la mémoire. Les sessions statistiques
sont des plages fixes UTC 00–08, 08–16 et 16–24, pas des horaires de places
financières corrigés du DST.

Le nettoyage est uniquement explicite :

```bash
.venv/bin/python main.py clean --pair EURUSD --timeframe M1
```

Il filtre uniquement les prix invalides et bornes OHLC incohérentes,
conserve les doublons ambigus, crée `data/cleaned/` et un CSV d'audit
par partition dans `data/reports/cleaning/`. Il ne touche ni aux archives
ni aux données traitées originales et ne remplit aucun trou.

### Export et backtest existant

```bash
.venv/bin/python main.py export --pair EURUSD --timeframe M5 \
  --side bid --format csv --output data/exports/EURUSD_M5_bid.csv
.venv/bin/python -m institutional --csv data/exports/EURUSD_M5_bid.csv \
  --pair EURUSD --price-basis bid --output reports/dukascopy_backtest.json
```

Choisir explicitement un côté pour éviter deux lignes au même timestamp.
L'export conserve les colonnes de provenance et écrit par partition.
Le backtest actuel utilise ses propres conventions de spread/exécution :
l'import Bid ne lui ajoute pas une simulation Tick Bid/Ask ni le spread
historique exact. Les volumes M1 exportés sont indicatifs.

### Dashboard et tests

```bash
.venv/bin/python -m streamlit run forex_historical_engine/dashboard.py
.venv/bin/python -B -m pytest tests/test_forex_historical_engine.py -q
DUKASCOPY_INTEGRATION=1 .venv/bin/python -B -m pytest \
  tests/test_forex_historical_engine.py -k real_m1 -q
.venv/bin/python -B -m unittest discover -s tests -v
```

Le dashboard permet de choisir paires/timeframes/dates, télécharger/reprendre,
suivre la progression, consulter les partitions, rapports, statuts et logs,
et préparer un export CSV sur disque. Il est destiné à une utilisation
locale mono-utilisateur.
Les tests unitaires utilisent exclusivement des données fictives et des
réponses HTTP simulées. Le test réel est désactivé par défaut et vérifie
EUR/USD M1 Bid/Ask, M5 reconstruit et une heure de ticks.

**État de la validation réelle :** les sondes initiales des archives du
6 janvier 2025 ont répondu HTTP 200 et permis de vérifier leurs formats.
La tentative d'intégration de bout en bout a ensuite reçu HTTP 429 sur
l'archive M1 Bid et a été interrompue. Aucun échantillon de marché n'a donc
été conservé et aucun succès d'intégration réelle n'est revendiqué.
La récupération effective nécessite un accès autorisé aux archives.
