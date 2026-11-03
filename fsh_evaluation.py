"""Kennzahlen, Verdikt-Logik und Sweeps (Komposition der Regeln aus `fsh_algorithm.py`; Verdikte sind reine Funktionen)."""

from dataclasses import dataclass

import numpy as np

import fsh_algorithm as A
import fsh_constants as C
from fsh_scenario import Market, hand_market, lie_market, make_market, market_from_rng

TIE_STEPS = 20000        # Schritte des zweiten Proportional-Response-Laufs aus zufälligem Start (Gleichstandsvergleich)


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    budgets: str = C.DEFAULT_BUDGETS
    seed: int = C.DEFAULT_SEED
    steps: int = C.DEFAULT_STEPS
    eta: float = C.DEFAULT_ETA


@dataclass(frozen=True, eq=False)
class Analysis:
    market: Market
    eq: A.Equilibrium
    gaps: dict
    nash_beaten: int
    rank: int                  # unabhängige Kreise des Trägers (0 = Wald = Zuteilung eindeutig)
    unique: bool               # Zuteilung zu diesen Preisen eindeutig (auch bei einem Kreis möglich, wenn er festgelegt ist)
    errs: dict                # Verfahren -> Preisfehler je Schritt
    work: dict                 # Verfahren -> Arbeit (Schritte bzw. Funktionsauswertungen) je Schritt

    @property
    def worst_gap(self) -> float:
        return max(abs(v) for v in self.gaps.values())

    def final(self, key: str) -> float:
        return float(self.errs[key][-1])

    def reached(self, key: str, target: float) -> int | None:
        """Arbeit (Schritte bzw. Auswertungen), bei der der Preisfehler erstmals <= target ist; None, wenn nie."""
        return steps_to(self.errs[key], self.work[key], target)


def steps_to(errs: np.ndarray, work: np.ndarray, target: float) -> int | None:
    hit = np.flatnonzero(errs <= target)
    return int(work[hit[0]]) if len(hit) else None


def analyse(settings: Settings) -> Analysis:
    mk = make_market(settings.kind, settings.n, settings.m, settings.budgets, settings.seed)
    eq = A.solve(mk.v, mk.budgets)
    x = eq.x
    g = A.gaps(mk.v, mk.budgets, eq.p, x)
    beaten = A.nash_beaten(mk.v, mk.budgets, x, np.random.default_rng(settings.seed + 3))
    tight = A.tight_edges(mk.v, eq.p)
    rank = A.cycle_rank(tight)
    unique = A.unique_allocation(tight, eq.spend, mk.budgets)
    steps = settings.steps
    e_pr, _ = A.run_pr(mk.v, mk.budgets, steps, eq.p)
    e_tat, _ = A.run_tatonnement(mk.v, mk.budgets, steps, eq.p, settings.eta)
    e_pg, ev_pg, _ = A.run_gradient(mk.v, mk.budgets, min(steps, C.PG_MAX_ITER), eq.p)
    ar = np.arange(1, steps + 1, dtype=float)
    return Analysis(mk, eq, g, beaten, rank, unique, {"pr": e_pr, "tat": e_tat, "pg": e_pg}, {"pr": ar, "tat": ar, "pg": ev_pg})


# --- Verdikte als reine Funktionen -------------------------------------------------------------------------------------------------------

def _num(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _sci(v: float) -> str:
    return f"{v:.1e}".replace(".", ",")


def _pct(v: float, d: int = 1) -> str:
    return _num(v, d)


def verdict_equilibrium(gaps: dict, exact: bool, nash_beaten: int, n: int, m: int) -> tuple[str, str]:
    """Kernfrage: Ist das ein Gleichgewicht? Alle Verletzungen unter 1e-9, Preise auf Rechengenauigkeit, keine zufällige Zuteilung mit höherer Nash-Wohlfahrt."""
    worst = max(abs(v) for v in gaps.values())
    if not exact:
        return "warning", f"Die Referenzlösung konnte nicht auf Rechengenauigkeit poliert werden (größte Abweichung {_sci(worst)}): Die Prüfer sind hier nur auf diese Genauigkeit zu lesen."
    if worst > 1e-9 or nash_beaten:
        return "error", f"Die Gleichgewichtsbedingungen sind verletzt (größte Abweichung {_sci(worst)}, {nash_beaten} Zufallszuteilungen mit höherer Nash-Wohlfahrt)."
    return "success", (f"Gleichgewicht: alle {m} Güter sind ausverkauft, alle {n} Budgets ausgegeben, jeder Käufer kauft nur das Beste für sein Geld (größte Abweichung {_sci(worst)}). "
                       f"Kein Käufer beneidet einen anderen, wenn man die Budgets einrechnet, und keine von {C.NASH_TRIALS} Zufallszuteilungen hat eine höhere Nash-Wohlfahrt.")


def verdict_methods(steps: int, final: dict, reached: dict) -> tuple[str, str]:
    """Drei Verfahren, ein Markt: Preisfehler nach `steps` Schritten und Arbeit bis 10^-6 (None = nie)."""
    parts = [f"{C.METHOD_LABEL[k]} {_sci(final[k])}" for k in ("pr", "tat", "pg")]
    head = f"Nach {steps} Schritten (Gradientenverfahren: höchstens {C.PG_MAX_ITER}) liegt der relative Preisfehler bei " + ", ".join(parts) + "."
    tail = []
    for k, what in (("pr", "Schritten"), ("tat", "Schritten"), ("pg", "Funktionsauswertungen")):
        r = reached[k]
        tail.append(f"{C.METHOD_LABEL[k]} erreicht 10⁻⁶ " + (f"nach {r} {what}" if r is not None else "nicht"))
    best = min(final, key=lambda k: final[k])
    kind = "info" if final["tat"] > 1e-6 or best != "tat" else "success"
    return kind, head + " " + "; ".join(tail) + f". Am genauesten ist hier: **{C.METHOD_LABEL[best]}**."


def verdict_ties(rank: int, unique: bool, price_diff: float | None, alloc_diff: float | None) -> tuple[str, str]:
    """Gleichstände: Ein Kreis im Träger lässt meist mehrere Zuteilungen zu; ob es zu den Preisen wirklich nur eine gibt, entscheidet `unique` (jede Ausgabe nimmt nur einen Wert an).
    Zwei Läufe aus verschiedenen Startpunkten haben dieselben Preise und, bei mehreren Zuteilungen, andere Zuteilungen."""
    if rank == 0:
        return "info", "Kein Gleichstand im Träger (der Käufer-Gut-Graph der besten Güter ist ein Wald): Die Zuteilung ist eindeutig, es gibt nur ein Gleichgewicht."
    word = "einen Kreis" if rank == 1 else f"{rank} unabhängige Kreise"
    if unique:
        return "info", (f"Der Träger enthält {word}, aber die Zuteilung ist trotzdem eindeutig: Die Summen der Budgets und Preise legen jede Ausgabe fest (Randfall). "
                        f"Es gibt nur ein Gleichgewicht, die Preise sind ohnehin eindeutig.")
    return "warning", (f"Der Träger enthält {word}, und es gibt mehrere Zuteilungen, die dieselben Preise tragen (die Preise sind trotzdem eindeutig). Zwei Läufe aus verschiedenen Startpunkten enden bei Preisen, "
                       f"die sich um {_sci(price_diff)} unterscheiden, aber bei Zuteilungen, die sich um bis zu {_num(alloc_diff)} je Eintrag unterscheiden.")


def verdict_lie(honest: float, best: float, best_ratio: float, true_ratio: float) -> tuple[str, str]:
    """Falschangabe im Handbeispiel: ehrlicher Nutzen gegen den besten gemeldeten (Verhältnis Gut 2 : Gut 1)."""
    if best <= honest + 1e-9:
        return "success", f"Hier lohnt keine Falschangabe: Ehrlichkeit (Verhältnis {_num(true_ratio)}) ist so gut wie jede gemeldete Alternative (Nutzen {_num(honest, 3)})."
    return "warning", (f"Ehrlich (Verhältnis {_num(true_ratio)}) erhält der Käufer den Nutzen {_num(honest, 3)}. Meldet er das Verhältnis {_num(best_ratio)}, steigt sein wahrer Nutzen auf "
                       f"**{_num(best, 3)}** (+{_pct(100 * (best / honest - 1))} %): Der Markt ist nicht wahrheitsgetreu.")


def verdict_search(wins: int, markets: int, median: float, q90: float, maximum: float) -> tuple[str, str]:
    """Suche nach Falschangaben in Zufallsmärkten: in wie vielen Märkten gewinnt Käufer 1, und wie viel."""
    if wins == 0:
        return "success", f"In keinem der {markets} Märkte gewinnt Käufer 1 durch eine der gesuchten Falschangaben."
    return "warning", (f"In **{wins} von {markets}** Märkten findet Käufer 1 in der Suche eine Falschangabe, die seinen wahren Nutzen erhöht (mindestens so viele Märkte); Gewinn im Median {_pct(100 * median)} %, 90-%-Quantil {_pct(100 * q90)} %, "
                       f"Maximum {_pct(100 * maximum)} % (über alle Märkte, auch die ohne Gewinn).")


# --- Handbeispiele, Gleichstände, Falschangaben ------------------------------------------------------------------------------------------

def hand_rows() -> list[dict]:
    """Die vier Handbeispiele mit Gleichgewichtspreisen und Nutzen."""
    rows = []
    for key in sorted(C.HAND):
        mk = hand_market(key)
        eq = A.solve(mk.v, mk.budgets)
        rows.append({"key": key, "name": C.HAND[key][0], "p": eq.p, "u": A.utilities(mk.v, eq.x), "x": eq.x})
    return rows


def tie_pair(market: Market, eq: A.Equilibrium, seed: int, steps: int = TIE_STEPS):
    """Zweiter Lauf von Proportional Response aus zufälligem Start: (Zuteilung, Preisabweichung zum Gleichgewicht, größter Unterschied der Zuteilungen je Eintrag)."""
    rng = np.random.default_rng(seed + 11)
    b0 = rng.random(market.v.shape) * (market.v > 0) + 1e-3 * (market.v > 0)
    b0 = b0 / b0.sum(axis=1, keepdims=True) * market.budgets[:, None]
    _, b = A.run_pr(market.v, market.budgets, steps, eq.p, b0)
    x2 = b / b.sum(axis=0)
    return x2, A.price_error(b.sum(axis=0), eq.p), float(np.abs(x2 - eq.x).max())


def tie_census(kind: str, mode: str = C.DEFAULT_BUDGETS, markets: int = C.TIE_CENSUS_MARKETS, size=C.TIE_CENSUS_SIZE) -> int:
    """Zahl der Märkte (von `markets`), deren Träger einen Kreis enthält (meist mehrere Zuteilungen; `tie_census_multiple` zählt die mit wirklich mehreren)."""
    return tie_census_detail(kind, mode, markets, size)[0]


def tie_census_multiple(kind: str, mode: str = C.DEFAULT_BUDGETS, markets: int = C.TIE_CENSUS_MARKETS, size=C.TIE_CENSUS_SIZE) -> int:
    """Zahl der Märkte (von `markets`) mit mehreren Zuteilungen zu den Gleichgewichtspreisen."""
    return tie_census_detail(kind, mode, markets, size)[1]


def tie_census_detail(kind: str, mode: str = C.DEFAULT_BUDGETS, markets: int = C.TIE_CENSUS_MARKETS, size=C.TIE_CENSUS_SIZE) -> tuple[int, int]:
    """(Märkte mit Kreis im Träger, davon mit mehreren Zuteilungen) über `markets` Märkte."""
    rng = np.random.default_rng(31)
    cycles = multiple = 0
    for _ in range(markets):
        mk = market_from_rng(kind, mode, size[0], size[1], rng)
        eq = A.solve(mk.v, mk.budgets)
        tight = A.tight_edges(mk.v, eq.p)
        has_cycle = A.cycle_rank(tight) > 0
        cycles += has_cycle
        multiple += has_cycle and not A.unique_allocation(tight, eq.spend, mk.budgets)
    return int(cycles), int(multiple)


def lie_curve(ratios=C.MANIP_RATIOS) -> tuple[list[float], float, float]:
    """Handbeispiel: Käufer 1 meldet (1, r) statt seiner Nutzen; Rückgabe (wahrer Nutzen je r, ehrlicher Nutzen, ehrliches Verhältnis)."""
    mk = lie_market()
    honest = float(mk.v[0] @ A.solve(mk.v, mk.budgets).x[0])
    us = [A.true_utility_after_report(mk.v, mk.budgets, 0, np.array([1.0, r])) for r in ratios]
    return us, honest, float(mk.v[0, 1] / mk.v[0, 0])


def manipulation_search(markets: int = C.MANIP_MARKETS, tries: int = C.MANIP_TRIES, size=C.MANIP_SIZE) -> dict:
    """Käufer 1 probiert `tries` Falschangaben (Streuung um den wahren Vektor oder Konzentration auf einen Teil der Güter); Gewinn = bester wahrer Nutzen / ehrlicher - 1."""
    rng = np.random.default_rng(13)
    gains = []
    overall = {"gain": 0.0, "market": None, "report": None}
    for _ in range(markets):
        mk = market_from_rng("stetig", C.DEFAULT_BUDGETS, size[0], size[1], rng)
        honest = float(mk.v[0] @ A.solve(mk.v, mk.budgets).x[0])
        best, best_report = 0.0, None
        for _ in range(tries):
            if rng.random() < 0.5:
                report = mk.v[0] * np.exp(rng.normal(0.0, 0.7, mk.m))
            else:
                report = mk.v[0] * (rng.random(mk.m) < 0.5) + 1e-3
            u = A.true_utility_after_report(mk.v, mk.budgets, 0, report)
            if np.isfinite(u) and u / honest - 1.0 > best:
                best, best_report = u / honest - 1.0, report
        gains.append(best)
        if best > overall["gain"]:
            overall = {"gain": best, "market": mk, "report": best_report}
    g = np.array(gains)
    return {"wins": int((g > C.MANIP_GAIN_TOL).sum()), "markets": markets, "median": float(np.median(g)), "q90": float(np.quantile(g, 0.9)), "max": float(g.max()), "best": overall}


# --- Vergleich der Verfahren über viele Märkte (Knopf) -----------------------------------------------------------------------------------

def _first(errs: np.ndarray, work, target: float) -> int | None:
    hit = np.flatnonzero(errs <= target)
    return int(work[hit[0]]) if len(hit) else None


def methods_table(sizes=C.TABLE_SIZES, seeds=C.TABLE_SEEDS, kind: str = "stetig") -> list[dict]:
    """Arbeit bis zum Preisfehler 10^-3 / 10^-6 / 10^-9 je Verfahren, Median über die Märkte (Anteil der Märkte, die das Ziel erreichen). Proportional Response und
    Tâtonnement in Schritten, das Gradientenverfahren in Funktionsauswertungen; beim Gradientenverfahren zählt nur, was über PG_NOISE_FLOOR liegt."""
    out = []
    for n, m in sizes:
        res = {k: {t: [] for t in C.TARGETS} for k in ("pr", "tat", "pg")}
        for seed in seeds:
            mk = make_market(kind, n, m, C.DEFAULT_BUDGETS, seed)
            eq = A.solve(mk.v, mk.budgets)
            e_pr, _ = A.run_pr(mk.v, mk.budgets, C.TABLE_PR_MAX, eq.p, stop_below=C.TARGETS[-1])
            e_tat, _ = A.run_tatonnement(mk.v, mk.budgets, C.TABLE_TAT_MAX, eq.p, C.DEFAULT_ETA, stop_below=C.TARGETS[-1])
            e_pg, ev_pg, _ = A.run_gradient(mk.v, mk.budgets, C.TABLE_PG_MAX, eq.p, stop_below=C.TARGETS[-1])
            for t in C.TARGETS:
                res["pr"][t].append(_first(e_pr, np.arange(1, len(e_pr) + 1), t))
                res["tat"][t].append(_first(e_tat, np.arange(1, len(e_tat) + 1), t))
                res["pg"][t].append(_first(e_pg, ev_pg, t))
        for t in C.TARGETS:
            row = {"n": n, "m": m, "target": t, "markets": len(seeds)}
            for k in ("pr", "tat", "pg"):
                if k == "pg" and t < C.PG_NOISE_FLOOR:
                    row[k] = {"median": None, "share": None}              # Rauschgrenze der Liniensuche: nicht berichtet
                    continue
                done = [v for v in res[k][t] if v is not None]
                row[k] = {"median": int(np.median(done)) if done else None, "share": len(done) / len(seeds)}
            out.append(row)
    return out
