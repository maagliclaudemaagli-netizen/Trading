# Protocole institutionnel : 2 zones invalidees / 2 zones creees

Le seul moteur actif est [institutional/](institutional/). Il ne contient pas
de filtre DBS, OTE, sweep, killzone, ATR, RR fixe ou nombre de mitigations.
Le depot contient ce moteur, le moteur de donnees Dukascopy, leurs
dependances, leur documentation et leurs tests de regression synthetiques.

## Utilisation

Dependances : `pandas`, `numpy`, `pyarrow` dans [requirements.txt](requirements.txt).

```bash
.venv/bin/python main.py download --pair EURUSD --timeframe M5 \
  --start 2025-01-06 --end 2025-01-06
.venv/bin/python main.py export --pair EURUSD --timeframe M5 \
  --side bid --output data/exports/EURUSD_M5_bid.csv
.venv/bin/python -m institutional --csv data/exports/EURUSD_M5_bid.csv \
  --pair EURUSD --price-basis bid \
  --output reports/institutional.json
```

Les donnees de marche ne sont pas incluses dans le depot versionne.
Le backtest lit un CSV local exporte depuis Dukascopy ; il ne declenche
aucun telechargement.
CSV : `timestamp,open,high,low,close,volume`, ou `time` en secondes Unix.
Les lignes doivent etre des M5 cloturees, triees, uniques, avec OHLC coherents.
`--end` est exclusif. Les dates sans fuseau sont interpretees en UTC.

## Architecture

- [market.py](institutional/market.py) : bougies, resampling, swings confirmes,
  FVG, OB, BB et registre des etats.
- [strategy.py](institutional/strategy.py) : tap, BOS, compteurs distincts,
  validation par cloture et signal avec trace des identifiants.
- [backtest.py](institutional/backtest.py) : ordres limites, spread, stops
  structurels, sorties, equity et metriques de performance.
- [__main__.py](institutional/__main__.py) : import CSV et rapports JSON/CSV.
- [tests/](tests/) : regression des metriques, du resampling et des exports.

## Hierarchies autorisees

| `--alignment` | Liquidite | Confirmation | Validation |
|---|---|---|---|
| intraday (defaut) | Daily | H1 | H4 |
| weekly | Weekly | H4 | Daily |
| swing | Monthly | Daily | Weekly |

Les registres H1/H4/Daily existent dans tous les cas. Le M5 est egalement
re-echantillonne en M15. Les jours commencent a 00:00 UTC, les semaines le lundi,
les mois le premier jour du calendrier.

## Sequence executee

1. Une M5 cloturee touche une zone macro FVG/OB/BB deja connue et non violee.
   Son sens determine celui du scenario. Aucune cloture Daily future n'est lue :
   toucher une ancienne zone Daily est observable avant la cloture du jour.
2. Une cloture de l'UT de confirmation casse son swing local dans ce sens.
3. Apres ce BOS, au moins **deux identifiants distincts** de zones opposees,
   qui existaient avant le tap, sont invalides par une cloture de confirmation.
4. Le meme mouvement cree au moins **deux identifiants distincts** de zones
   dans son sens. Les evenements de la bougie BOS peuvent compter ; ceux
   precedant le BOS ne comptent pas. Une attenuation n'est pas une invalidation.
5. La validation attend une bougie de l'UT prevue, directionnelle, cloturant
   strictement au-dela du niveau BOS. Une meche ou une egalite ne suffit pas.
   Selon la clarification donnee, c'est la **cloture** qui doit depasser le
   niveau, pas obligatoirement l'ouverture et la cloture toutes deux.
6. La cible est le prix exact du swing externe admissible dans la direction du
   trade. L'ordre attend ensuite un repli sur une zone creee par ce mouvement.

Les regles sont symetriques pour les shorts. Un BOS oppose met fin au mouvement ;
une violation de la zone macro annule le scenario. Aucun delai d'expiration
arbitraire n'est applique. Un scenario par sens peut attendre sa confirmation ;
une fois un ordre emis, une seule position ou un seul ordre est autorise.

## Definitions operationnelles explicites

La source ne definit pas numeriquement tous ses termes. Ces conventions
concretisent les details dont le choix a ete delegue par l'utilisateur ; elles
ne sont pas presentees comme des citations de la source.

- Swing : extremum strict avec deux bougies de chaque cote, connu seulement
  apres la cloture des deux bougies de droite.
- FVG : trois bougies successives ; haussier si `low[2] > high[0]`, baissier
  si `high[2] < low[0]`. Bornes exactes de ces meches, sans seuil de taille.
- OB : derniere bougie de corps oppose, du pivot casse jusqu'avant la bougie
  BOS ; bornes high/low. Un meme OB n'est pas recree plusieurs fois.
- BB : rupture en cloture du bord oppose d'un OB ; l'OB devient viole et un
  BB de sens inverse apparait avec les memes bornes. Pas de conversion recursive.
- Etats : actif, attenue au premier chevauchement observe, viole uniquement
  apres cloture de son UT au-dela du bord oppose. Une zone attenuee reste
  admissible. Les identifiants sont propres au type, au sens, a l'UT et a
  l'origine ; deux types superposes sont deux zones, une zone repetee n'en est qu'une.
- Si plusieurs zones macro de meme sens sont touchees, la plus recente sert
  de reference. Le comptage s'effectue sur l'UT de confirmation, jamais en
  additionnant indistinctement les registres de plusieurs UT.
- Creux/sommet d'impulsion : extreme des M5 depuis le tap jusqu'a la seconde
  invalidation opposee ; fige ensuite. Le stop long est sous ce creux, le stop
  short au-dessus de ce sommet, avec le demi-spread approprie.
- Entree : milieu de la plus recente zone creee encore valide, situee du cote
  du repli par rapport a la cloture de validation. Departages deterministes.
- Liquidite externe : swing dans le sens du trade, confirme avant le tap sur
  l'UT macro ou de validation, encore intact et au-dela de la cloture de
  validation ; le plus proche est retenu. La consommation est observee des
  la cloture M5, meme si la Daily n'est pas encore terminee.

## Causalite et simulation

Le resampling est labelle/ferme a gauche, puis **`shift(1)` sur la grille
calendaire complete**, avant retrait des cases vides. Une H4 08:00-12:00
n'est accessible qu'a 12:00. Une borne finale preserve la derniere bougie
complete, sans admettre une bougie partielle. Aucun trou de cotation n'est rempli.
La qualite/completude des bougies source reste a verifier aupres du fournisseur.
Les registres sont mis a jour a chaque cloture ; les compteurs H1 sont connus
avant une validation H4 simultanee, jamais avant cette heure.

Spread total fixe : **0.6 pip**, pip de 0.01 pour JPY, 0.0001 sinon.
`--price-basis mid` est l'hypothese par defaut ; `bid` convertit les cotations
avec ce spread. Aucune commission ni aucun slippage supplementaire ne sont ajoutes.
Entrees long au ask, short au bid ; sorties sur le cote oppose.
Le TP conserve exactement le prix numerique du swing externe, sans RR fixe.
Le risque utilise 1 % du capital realise et la distance prevue entre limite
et stop, spread compris. Capital de depart : 10 000.

Un gap au stop sort au prix d'ouverture defavorable. Une limite TP sort a son
prix exact. Si stop et TP sont touches dans la meme M5, le stop est prioritaire.
Un TP dans la bougie d'une entree limite intrabar n'est pas credite, faute de
connaitre l'ordre des ticks. Ces conventions prudentes ne pretendent pas
reconstituer exactement un chemin intrabar absent des donnees OHLC.
Les heures exportees sont les bornes des bougies d'execution, pas des ticks.
Les positions ouvertes a la fin ne sont pas fermees artificiellement.

## Rapports d'execution

Le JSON et ses CSV `.trades.csv` / `.audit.csv` exposent les resultats et les
identifiants justifiant chaque confirmation. Les metriques comprennent
esperance R, taux de reussite, PF, drawdown realise depuis le capital initial,
drawdown marque au marche, positions/ordres encore ouverts. `null` indique une
mesure non definie ; `"Infinity"` un PF avec gains et aucune perte.

Les rapports sont generes a partir des donnees fournies a chaque execution.
Leur dossier de sortie est cree si necessaire. Les repertoires locaux
`data/` et `reports/` sont exclus de Git ; aucun historique de resultats ni
donnees de marche ne sont inclus dans le depot.
