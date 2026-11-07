"""MTF SMC v2 — 4H Market Structure Shift -> 15M trigger, moteur numba + Optuna + WFA + Monte Carlo.

Différences vs mtf_smc_backtest.py (v1)
---------------------------------------
* Biais 4H = Market Structure Shift : clôture 4H au-delà du dernier swing high/low
  CONFIRMÉ (fractal N barres de chaque côté -> connu seulement N barres après le pivot).
* Zones 4H = OB / FVG de la jambe d'impulsion qui a cassé la structure (pas n'importe quel
  gap/OB). Zones opposées supprimées au MSS inverse, invalidées par clôture 4H, TTL.
* Killzones (heure UTC de l'ouverture de la bougie d'entrée) : London 07-10, NY 12-15.
* SL structurel : swing extrême des `sl_lb` dernières 15M - atr_buf*ATR - demi-spread,
  avec plancher de risque min_risk_atr*ATR (évite les stops de bruit).
* Exécution bid/ask simulée à partir du mid : entrée au ask/bid + slippage, stop déclenché
  sur le côté bid (long) / ask (short), fill au pire de (stop, open) en cas de gap + slippage,
  TP en limite sans slippage, SL prioritaire si SL et TP dans la même bougie, time-stop.
* Pipeline : --mode default | optimize | wfa (Walk-Forward ré-optimisé par Optuna + Monte Carlo).

Usage:
    .venv/bin/python mtf_smc_v2.py --mode default
    .venv/bin/python mtf_smc_v2.py --mode optimize --trials 200 --end 2023-01-01
    .venv/bin/python mtf_smc_v2.py --mode wfa --train-months 24 --test-months 6 --trials 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from numba import njit

DATA = Path(__file__).parent / "data"
PIP = 0.0001
CAP = 256  # capacité du buffer de zones

# spread moyen typique (pips) en sessions liquides, broker ECN/raw + commission incluse
SPREADS = {"EURUSD": 0.4, "GBPUSD": 0.7, "USDJPY": 0.5, "AUDUSD": 0.5, "USDCAD": 0.8,
           "USDCHF": 0.8, "NZDUSD": 0.9, "EURJPY": 0.8, "GBPJPY": 1.5}


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001

KILLZONES = {
    "LDN": range(7, 10),
    "NY": range(12, 15),
    "BOTH": [*range(7, 10), *range(12, 15)],
    "ALL": range(24),
}

DEFAULT = dict(
    swing_n=3, zones="BOTH", trigger=1, sweep_lb=12, sl_lb=6, atr_buf=0.3,
    min_risk_atr=1.0, rr=2.5, max_hold=96, zone_ttl=30, killzone="BOTH",
)


# =========================================================== données
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx
    return df.sort_index()[["open", "high", "low", "close"]].astype(float)


class Market:
    """Arrays pré-calculés partagés par tous les backtests (une seule fois)."""

    def __init__(self, pair: str, spread_pips: float, slip_pips: float):
        m15 = load(DATA / f"{pair}_15M.csv")
        h4 = load(DATA / f"{pair}_4H.csv")
        # on retire la dernière 15M si potentiellement incomplète (heure de génération du CSV)
        m15 = m15.iloc[:-1]
        self.index = m15.index
        self.o, self.h, self.l, self.c = (m15[k].to_numpy() for k in ("open", "high", "low", "close"))
        prev_c = np.roll(self.c, 1)
        prev_c[0] = self.c[0]
        tr = np.maximum(self.h - self.l, np.maximum(abs(self.h - prev_c), abs(self.l - prev_c)))
        self.atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        self.hour = self.index.hour.to_numpy().astype(np.int64)
        t = self.index.as_unit("ns").asi8
        self.contiguous = np.append(np.diff(t) == 15 * 60 * 10**9, False)
        # 4H : bougie k utilisable seulement quand son heure de clôture <= ouverture de la 15M
        close4 = (h4.index + pd.Timedelta(hours=4)).as_unit("ns").asi8
        self.h4_avail = np.searchsorted(close4, t, side="right").astype(np.int64)
        self.o4, self.h4, self.l4, self.c4 = (h4[k].to_numpy() for k in ("open", "high", "low", "close"))
        pip = pip_size(pair)
        self.pair = pair
        self.hs = spread_pips * pip / 2
        self.slip = slip_pips * pip

    def bounds(self, start=None, end=None) -> tuple[int, int]:
        j0 = 0 if start is None else int(self.index.searchsorted(pd.Timestamp(start, tz="UTC")))
        j1 = len(self.index) if end is None else int(self.index.searchsorted(pd.Timestamp(end, tz="UTC")))
        return j0, j1


# =========================================================== moteur numba
@njit(cache=True)
def _run(o, h, l, c, atr, hour, contiguous, h4_avail, o4, h4, l4, c4, j0, j1,
         swing_n, use_ob, use_fvg, trigger, sweep_lb, sl_lb, atr_buf, min_risk_atr,
         rr, max_hold, zone_ttl, kz, hs, slip):
    n = o.shape[0]
    out_e = np.empty(n // 4, np.int64)
    out_x = np.empty(n // 4, np.int64)
    out_d = np.empty(n // 4, np.int64)
    out_r = np.empty(n // 4, np.float64)
    nt = 0
    # zones
    zt = np.zeros(CAP); zb = np.zeros(CAP)
    zs = np.zeros(CAP, np.int64); zborn = np.zeros(CAP, np.int64)
    zon = np.zeros(CAP, np.bool_)
    zc = 0
    # structure 4H
    sh_p = 0.0; sh_i = -1; sh_used = True
    sl_p = 0.0; sl_i = -1; sl_used = True
    bias = 0
    k_done = 0
    # position
    pos = 0; ent = 0.0; stop = 0.0; tgt = 0.0; risk = 0.0; e_bar = 0

    for j in range(1, n - 1):
        # ---------- intégration des bougies 4H clôturées
        while k_done < h4_avail[j]:
            k = k_done
            p = k - swing_n
            if p >= swing_n:
                is_h = True; is_l = True
                for q in range(p - swing_n, p + swing_n + 1):
                    if q != p:
                        if h4[q] >= h4[p]:
                            is_h = False
                        if l4[q] <= l4[p]:
                            is_l = False
                if is_h:
                    sh_p = h4[p]; sh_i = p; sh_used = False
                if is_l:
                    sl_p = l4[p]; sl_i = p; sl_used = False
            for d in (1, -1):
                broke = False
                start = 0
                if d == 1 and (not sh_used) and c4[k] > sh_p:
                    sh_used = True; broke = True; start = sl_i if sl_i >= 0 else k - 10
                if d == -1 and (not sl_used) and c4[k] < sl_p:
                    sl_used = True; broke = True; start = sh_i if sh_i >= 0 else k - 10
                if not broke:
                    continue
                bias = d
                start = max(start, 2)
                for z in range(CAP):
                    if zon[z] and zs[z] != d:
                        zon[z] = False
                if use_ob:
                    for m in range(k, start - 1, -1):
                        if (d == 1 and c4[m] < o4[m]) or (d == -1 and c4[m] > o4[m]):
                            zb[zc] = l4[m]; zt[zc] = h4[m]; zs[zc] = d; zborn[zc] = k; zon[zc] = True
                            zc = (zc + 1) % CAP
                            break
                if use_fvg:
                    for m in range(start + 2, k + 1):
                        bot = 0.0
                        top = 0.0
                        if d == 1 and l4[m] > h4[m - 2]:
                            bot = h4[m - 2]; top = l4[m]
                            for q in range(m + 1, k + 1):
                                top = min(top, l4[q])
                        elif d == -1 and h4[m] < l4[m - 2]:
                            bot = h4[m]; top = l4[m - 2]
                            for q in range(m + 1, k + 1):
                                bot = max(bot, h4[q])
                        else:
                            continue
                        if top > bot:
                            zb[zc] = bot; zt[zc] = top; zs[zc] = d; zborn[zc] = k; zon[zc] = True
                            zc = (zc + 1) % CAP
            for z in range(CAP):
                if zon[z]:
                    if k - zborn[z] > zone_ttl or (zs[z] == 1 and c4[k] < zb[z]) or (zs[z] == -1 and c4[k] > zt[z]):
                        zon[z] = False
            k_done += 1

        # ---------- gestion de la position sur la bougie j
        if pos != 0:
            done = False; fill = 0.0
            if pos == 1:
                bo = o[j] - hs; bl = l[j] - hs; bh = h[j] - hs
                if bl <= stop:
                    fill = min(bo, stop) - slip; done = True
                elif bh >= tgt:
                    fill = max(bo, tgt) if j > e_bar else tgt; done = True
            else:
                ao = o[j] + hs; ah = h[j] + hs; al = l[j] + hs
                if ah >= stop:
                    fill = max(ao, stop) + slip; done = True
                elif al <= tgt:
                    fill = min(ao, tgt) if j > e_bar else tgt; done = True
            if not done and j - e_bar + 1 >= max_hold:
                fill = c[j] - pos * (hs + slip); done = True
            if done:
                out_e[nt] = e_bar; out_x[nt] = j; out_d[nt] = pos
                out_r[nt] = (fill - ent) * pos / risk
                nt += 1
                pos = 0
            continue  # pas de nouveau signal sur une bougie où l'on était en position

        # ---------- signal sur la bougie j clôturée -> entrée à l'open de j+1
        if bias == 0 or j + 1 < j0 or j + 1 >= j1 or not contiguous[j] or not kz[hour[j + 1]]:
            continue
        if j < max(sweep_lb, sl_lb) + 2:
            continue
        d = bias
        if d == 1:
            engulf = c[j - 1] < o[j - 1] and c[j] > o[j] and c[j] >= o[j - 1] and o[j] <= c[j - 1]
            ref = l[j - 2]
            for q in range(j - 1 - sweep_lb, j - 1):
                ref = min(ref, l[q])
            swept = min(l[j], l[j - 1]) < ref and c[j] > ref
            body_ok = c[j] > o[j]
        else:
            engulf = c[j - 1] > o[j - 1] and c[j] < o[j] and c[j] <= o[j - 1] and o[j] >= c[j - 1]
            ref = h[j - 2]
            for q in range(j - 1 - sweep_lb, j - 1):
                ref = max(ref, h[q])
            swept = max(h[j], h[j - 1]) > ref and c[j] < ref
            body_ok = c[j] < o[j]
        if trigger == 0:
            ok = engulf
        elif trigger == 1:
            ok = engulf and swept
        else:
            ok = swept and body_ok
        if not ok:
            continue
        zbest = -1
        lo2 = min(l[j], l[j - 1]); hi2 = max(h[j], h[j - 1])
        for z in range(CAP):
            if zon[z] and zs[z] == d and lo2 <= zt[z] and hi2 >= zb[z]:
                if zbest < 0 or zborn[z] > zborn[zbest]:
                    zbest = z
        if zbest < 0:
            continue
        a = atr[j]
        if d == 1:
            ent = o[j + 1] + hs + slip
            sw = l[j]
            for q in range(j - sl_lb + 1, j + 1):
                sw = min(sw, l[q])
            stop = min(sw - atr_buf * a - hs, ent - min_risk_atr * a)
        else:
            ent = o[j + 1] - hs - slip
            sw = h[j]
            for q in range(j - sl_lb + 1, j + 1):
                sw = max(sw, h[q])
            stop = max(sw + atr_buf * a + hs, ent + min_risk_atr * a)
        risk = (ent - stop) * d
        if risk <= 0:
            continue
        tgt = ent + d * rr * risk
        pos = d; e_bar = j + 1
        zon[zbest] = False  # zone consommée
    return out_e[:nt], out_x[:nt], out_d[:nt], out_r[:nt]


def backtest(mk: Market, p: dict, start=None, end=None) -> pd.DataFrame:
    j0, j1 = mk.bounds(start, end)
    kz = np.zeros(24, np.bool_)
    kz[list(KILLZONES[p["killzone"]])] = True
    e, x, d, r = _run(
        mk.o, mk.h, mk.l, mk.c, mk.atr, mk.hour, mk.contiguous, mk.h4_avail,
        mk.o4, mk.h4, mk.l4, mk.c4, j0, j1,
        int(p["swing_n"]), p["zones"] in ("OB", "BOTH"), p["zones"] in ("FVG", "BOTH"),
        int(p["trigger"]), int(p["sweep_lb"]), int(p["sl_lb"]), float(p["atr_buf"]),
        float(p["min_risk_atr"]), float(p["rr"]), int(p["max_hold"]), int(p["zone_ttl"]),
        kz, mk.hs, mk.slip,
    )
    return pd.DataFrame({"entry_time": mk.index[e], "exit_time": mk.index[x], "dir": d, "R": r})


# =========================================================== métriques
def metrics(r: np.ndarray, risk_pct: float = 1.0) -> dict:
    r = np.asarray(r, float)
    if r.size == 0:
        return {"Trades": 0}
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumprod(1 + r * risk_pct / 100)
    sd = r.std(ddof=1) if r.size > 1 else np.nan
    return {
        "Trades": r.size,
        "Win Rate %": round((r > 0).mean() * 100, 1),
        "Profit Factor": round(gains / losses, 2) if losses else np.inf,
        "Expectancy R": round(r.mean(), 3),
        "t-stat": round(r.mean() / sd * np.sqrt(r.size), 2) if sd and sd > 0 else np.nan,
        "Max DD %": round((1 - eq / np.maximum.accumulate(eq)).max() * 100, 1),
        "Return %": round((eq[-1] - 1) * 100, 1),
    }


def score(r: np.ndarray, min_trades: int) -> float:
    """t-stat de l'espérance (pénalise la variance et le faible nombre de trades)."""
    if r.size < min_trades:
        return -10.0 + r.size / min_trades
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(r.size)) if sd > 0 else -10.0


# =========================================================== Optuna
def suggest(trial: optuna.Trial) -> dict:
    return dict(
        swing_n=trial.suggest_int("swing_n", 2, 6),
        zones=trial.suggest_categorical("zones", ["OB", "FVG", "BOTH"]),
        trigger=trial.suggest_int("trigger", 0, 2),
        sweep_lb=trial.suggest_int("sweep_lb", 4, 32, step=4),
        sl_lb=trial.suggest_int("sl_lb", 2, 16),
        atr_buf=trial.suggest_float("atr_buf", 0.0, 1.0, step=0.1),
        min_risk_atr=trial.suggest_float("min_risk_atr", 0.5, 3.0, step=0.25),
        rr=trial.suggest_float("rr", 1.0, 4.0, step=0.25),
        max_hold=trial.suggest_int("max_hold", 16, 192, step=16),
        zone_ttl=trial.suggest_int("zone_ttl", 6, 60, step=6),
        killzone=trial.suggest_categorical("killzone", ["LDN", "NY", "BOTH", "ALL"]),
    )


def optimize(mk: Market, start, end, trials: int, min_trades: int, seed: int = 0) -> tuple[dict, float]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: score(backtest(mk, suggest(t), start, end).R.to_numpy(), min_trades),
                   n_trials=trials)
    return {**study.best_params}, study.best_value


# =========================================================== Monte Carlo
def monte_carlo(r: np.ndarray, n_sims: int = 10_000, risk_pct: float = 1.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    r = np.asarray(r, float)
    n = r.size
    idx = rng.integers(0, n, (n_sims, n))
    boot = r[idx].mean(axis=1)
    null = (r - r.mean())[idx].mean(axis=1)          # bootstrap sous H0 : E[R] = 0
    signs = rng.choice([-1.0, 1.0], (n_sims, n))      # test de permutation de signe
    perm = (np.abs(r) * signs).mean(axis=1)
    dds = np.empty(min(n_sims, 2000))
    for i in range(dds.size):                         # DD sous ré-ordonnancement des trades
        eq = np.cumprod(1 + rng.permutation(r) * risk_pct / 100)
        dds[i] = (1 - eq / np.maximum.accumulate(eq)).max() * 100
    return {
        "E[R] observé": round(r.mean(), 3),
        "IC95 E[R]": (round(np.percentile(boot, 2.5), 3), round(np.percentile(boot, 97.5), 3)),
        "p-value bootstrap (H0: E[R]<=0)": round((null >= r.mean()).mean(), 4),
        "p-value permutation de signe": round((perm >= r.mean()).mean(), 4),
        "P(E[R]>0)": round((boot > 0).mean(), 3),
        "Max DD % médian / p95": (round(np.median(dds), 1), round(np.percentile(dds, 95), 1)),
    }


# =========================================================== Walk-Forward
def walk_forward(mk: Market, train_m: int, test_m: int, trials: int, min_trades: int) -> pd.DataFrame:
    first = mk.index[0].normalize().tz_localize(None)
    last = mk.index[-1].tz_localize(None)
    folds, oos = [], []
    t0 = first
    while True:
        tr_end = t0 + pd.DateOffset(months=train_m)
        te_end = tr_end + pd.DateOffset(months=test_m)
        if tr_end >= last:
            break
        params, is_score = optimize(mk, t0, tr_end, trials, min_trades, seed=len(folds))
        is_r = backtest(mk, params, t0, tr_end).R.to_numpy()
        tr = backtest(mk, params, tr_end, min(te_end, last + pd.Timedelta(days=1)))
        oos.append(tr)
        m = metrics(tr.R.to_numpy())
        folds.append({"test": f"{tr_end:%Y-%m}→{te_end:%Y-%m}", "IS E[R]": round(is_r.mean(), 3) if is_r.size else np.nan,
                      "IS n": is_r.size, "OOS n": m["Trades"], "OOS E[R]": m.get("Expectancy R"),
                      "OOS PF": m.get("Profit Factor"), "params": params})
        print(f"  fold {folds[-1]['test']}: IS E[R]={folds[-1]['IS E[R]']} ({is_r.size})  "
              f"OOS E[R]={m.get('Expectancy R')} ({m['Trades']})", flush=True)
        t0 = t0 + pd.DateOffset(months=test_m)
    print("\n" + pd.DataFrame(folds).drop(columns="params").to_string(index=False))
    return pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=["R"])


# =========================================================== Multi-paires
def multi(args) -> None:
    """Pour chaque paire : (a) paramètres par défaut figés, (b) WFA ré-optimisé par paire.
    Puis agrégation de tous les trades OOS + Monte Carlo sur le portefeuille."""
    fixed_rows, wfa_rows, pooled_fixed, pooled_wfa = [], [], [], []
    for pair in args.pairs:
        mk = Market(pair, SPREADS.get(pair, args.spread), args.slip)
        fx = backtest(mk, DEFAULT).assign(pair=pair)
        pooled_fixed.append(fx)
        fixed_rows.append({"Paire": pair, "Spread": SPREADS.get(pair), **metrics(fx.R.to_numpy())})
        print(f"\n=== {pair} — WFA", flush=True)
        oos = walk_forward(mk, args.train_months, args.test_months, args.trials, args.min_trades).assign(pair=pair)
        pooled_wfa.append(oos)
        m = metrics(oos.R.to_numpy())
        mc = monte_carlo(oos.R.to_numpy()) if len(oos) > 5 else {}
        wfa_rows.append({"Paire": pair, **m, "p-value": mc.get("p-value bootstrap (H0: E[R]<=0)")})

    pd.set_option("display.width", 220)
    print("\n##### Paramètres par défaut FIGÉS (aucune optimisation) #####")
    print(pd.DataFrame(fixed_rows).to_string(index=False))
    print("\n##### Walk-Forward OOS (ré-optimisé par paire) #####")
    print(pd.DataFrame(wfa_rows).to_string(index=False))
    for name, frames in (("FIGÉ", pooled_fixed), ("WFA OOS", pooled_wfa)):
        allt = pd.concat(frames).sort_values("entry_time")
        r = allt.R.to_numpy()
        print(f"\n##### Portefeuille {name} ({len(args.pairs)} paires) #####")
        print(metrics(r))
        for k, v in monte_carlo(r).items():
            print(f"  {k}: {v}")
        allt.to_csv(DATA / f"mtf_smc_v2_multi_{name.split()[0].lower()}.csv", index=False)


# =========================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--mode", choices=["default", "optimize", "wfa", "multi"], default="default")
    ap.add_argument("--pairs", nargs="*", default=list(SPREADS))
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--min-trades", type=int, default=40)
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--spread", type=float, default=0.4, help="spread total en pips (bid/ask)")
    ap.add_argument("--slip", type=float, default=0.1, help="slippage en pips par fill marché/stop")
    args = ap.parse_args()

    if args.mode == "multi":
        multi(args)
        return
    mk = Market(args.pair, args.spread, args.slip)
    print(f"{args.pair} 15M {mk.index[0]:%Y-%m-%d} -> {mk.index[-1]:%Y-%m-%d} | spread {args.spread} + slip "
          f"{args.slip}x2 pips")

    if args.mode == "default":
        rows = []
        for name, over in {
            "Défaut (MSS+sweep+KZ)": {},
            "Sans killzone": {"killzone": "ALL"},
            "Trigger engulf seul": {"trigger": 0},
            "OB seul": {"zones": "OB"}, "FVG seul": {"zones": "FVG"},
            **{f"RR {rr}": {"rr": rr} for rr in (2, 3, 4)},
        }.items():
            tr = backtest(mk, {**DEFAULT, **over}, args.start, args.end)
            rows.append({"Variante": name, **metrics(tr.R.to_numpy())})
        print(pd.DataFrame(rows).to_string(index=False))
        return

    if args.mode == "optimize":
        params, best = optimize(mk, args.start, args.end, args.trials, args.min_trades)
        print(f"Meilleur score IS (t-stat) = {best:.2f}\n{params}")
        print("IS :", metrics(backtest(mk, params, args.start, args.end).R.to_numpy()))
        if args.end:
            oos = backtest(mk, params, args.end, None).R.to_numpy()
            print("OOS:", metrics(oos))
            if oos.size > 5:
                print("Monte Carlo OOS:", monte_carlo(oos))
        return

    oos = walk_forward(mk, args.train_months, args.test_months, args.trials, args.min_trades)
    r = oos.R.to_numpy()
    print("\nOOS concaténé :", metrics(r))
    if r.size > 5:
        for k, v in monte_carlo(r).items():
            print(f"  {k}: {v}")
    oos.to_csv(DATA / f"mtf_smc_v2_wfa_oos_{args.pair}.csv", index=False)


if __name__ == "__main__":
    main()
