"""Zeilen der README-Tabellen, gerechnet über dieselben Funktionen wie in der App (fsh_evaluation) und in der Rundung der App.

`tests/test_claims.py` prüft jede dieser Zeilen gegen die README; `python tools/readme_rows.py` druckt sie zum Einfügen.
Alle Zahlen sind deterministisch (feste Seeds)."""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fsh_algorithm as A  # noqa: E402
import fsh_constants as C  # noqa: E402
import fsh_evaluation as ev  # noqa: E402
from fsh_scenario import hand_market, make_market  # noqa: E402

CHECK_SEEDS = tuple(range(300000, 300020))
CHECK_SIZES = ((6, 5), (20, 10))
ETA_FACTORS = (0.25, 1.0, 2.0, 4.0, 8.0)
ETA_STEPS = 2000
ETA_SIZES = ((6, 5), (20, 10))


def c(v: float, d: int = 2) -> str:
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")


def sci(v: float) -> str:
    return f"{v:.1e}".replace(".", ",")


@lru_cache(maxsize=None)
def table_result() -> list[dict]:
    return ev.methods_table()


@lru_cache(maxsize=None)
def search_result() -> dict:
    return ev.manipulation_search()


def hand_rows() -> list[str]:
    rows = []
    for r in ev.hand_rows():
        v, b = C.HAND[r["key"]][1], C.HAND[r["key"]][2]
        vs = "; ".join("(" + ", ".join(f"{x:g}" for x in row).replace(".", ",") + ")" for row in v)
        rows.append(f"| **{r['key']}**: {r['name']} | {vs} | {', '.join(f'{x:g}' for x in b)} | {'; '.join(c(float(x), 3) for x in r['p'])} | {'; '.join(c(float(x), 3) for x in r['u'])} |")
    return rows


def check_rows() -> list[str]:
    """Gleichgewichtsprüfer: größte Abweichung über 20 Märkte je Erzeugerart und Größe; Märkte mit Kreis im Träger."""
    rows = []
    for kind in C.KINDS:
        for n, m in CHECK_SIZES:
            worst, cycles, beaten = 0.0, 0, 0
            for seed in CHECK_SEEDS:
                mk = make_market(kind, n, m, C.DEFAULT_BUDGETS, seed)
                eq = A.solve(mk.v, mk.budgets)
                g = A.gaps(mk.v, mk.budgets, eq.p, eq.x)
                worst = max(worst, max(abs(x) for x in g.values()))
                cycles += A.cycle_rank(A.tight_edges(mk.v, eq.p)) > 0
                beaten += A.nash_beaten(mk.v, mk.budgets, eq.x, np.random.default_rng(seed), trials=50)
            rows.append(f"| {C.KIND_LABEL[kind]} | {n}×{m} | {len(CHECK_SEEDS)} | {'unter 1e-9' if worst < 1e-9 else sci(worst)} | {beaten} | {cycles} von {len(CHECK_SEEDS)} |")
    return rows


def method_rows() -> list[str]:
    """Arbeit bis zum Preisfehler 10^-3 / 10^-6 / 10^-9 (Zeilen wie in der App-Tabelle)."""
    rows = []
    for r in table_result():
        def cell(k):
            cc = r[k]
            if cc["share"] is None:
                return "Rauschgrenze"
            return "nie" if cc["median"] is None else f"{cc['median']}" + ("" if cc["share"] == 1.0 else f" ({c(100 * cc['share'], 0)} %)")
        rows.append(f"| {r['n']}×{r['m']} | ≤ {sci(r['target'])} | {cell('pr')} | {cell('tat')} | {cell('pg')} |")
    return rows


def eta_rows() -> list[str]:
    """Tâtonnement: Preisfehler nach 2000 Schritten, Median über die Märkte, je Faktor c und Größe."""
    rows = []
    for c_eta in ETA_FACTORS:
        cells = []
        for n, m in ETA_SIZES:
            vals = []
            for seed in C.TABLE_SEEDS:
                mk = make_market("stetig", n, m, C.DEFAULT_BUDGETS, seed)
                eq = A.solve(mk.v, mk.budgets)
                errs, _ = A.run_tatonnement(mk.v, mk.budgets, ETA_STEPS, eq.p, c_eta)
                vals.append(errs[-1])
            lost = sum(v > 1.0 for v in vals)
            cells.append(f"über 100 % in {lost} von {len(vals)}" if lost >= len(vals) // 2 else sci(float(np.median(vals))))
        rows.append(f"| {c(c_eta)} | " + " | ".join(cells) + " |")
    return rows


def census_row() -> str:
    census = {k: ev.tie_census(k) for k in C.KINDS}
    return "| " + " | ".join(f"{census[k]} von {C.TIE_CENSUS_MARKETS}" for k in C.KINDS) + " |"


def tie_numbers() -> dict:
    """Standardmarkt mit ganzzahligen Nutzen: Kreise im Träger, Preisabweichung und größter Unterschied der Zuteilungen zwischen Referenz und zweitem PR-Lauf."""
    s = ev.Settings(kind="ganzzahlig")
    a = ev.analyse(s)
    _, price_diff, alloc_diff = ev.tie_pair(a.market, a.eq, s.seed)
    return {"rank": a.rank, "price_diff": price_diff, "alloc_diff": alloc_diff}


def lie_numbers() -> dict:
    us, honest, true_ratio = ev.lie_curve()
    best = int(np.argmax(us))
    return {"honest": honest, "best": us[best], "ratio": C.MANIP_RATIOS[best], "true_ratio": true_ratio}


def search_row() -> str:
    r = search_result()
    return f"| {r['wins']} von {r['markets']} | {c(100 * r['median'], 1)} % | {c(100 * r['q90'], 1)} % | {c(100 * r['max'], 1)} % |"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for name, rows in (("Handbeispiele", hand_rows()), ("Prüfer", check_rows()), ("Verfahren", method_rows()), ("Tâtonnement c", eta_rows())):
        print(f"\n{name}")
        print("\n".join(rows))
    print("\nZählung Gleichstände\n" + census_row())
    print("\nGleichstandsmarkt", tie_numbers())
    print("\nFalschangabe", lie_numbers())
    print("\nSuche\n" + search_row())
