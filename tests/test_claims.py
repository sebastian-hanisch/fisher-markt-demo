"""Jede Zahl der README steht hier und wird über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Tabellen der README werden Zeile für Zeile gegen `tools/readme_rows.py` geprüft (dieselbe Rundung wie die App); die einzelnen Sätze der README
haben je einen Test unter der Bedingung, die in der README daneben steht."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import fsh_algorithm as A  # noqa: E402
import fsh_constants as C  # noqa: E402
import fsh_evaluation as ev  # noqa: E402
import readme_rows as R  # noqa: E402
from fsh_scenario import make_market  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_tables_in_the_readme_are_the_computed_rows():
    for name, rows in (("Tabelle 1", R.hand_rows()), ("Tabelle 2", R.check_rows()), ("Tabelle 3", R.method_rows()), ("Tabelle 4", R.eta_rows())):
        for row in rows:
            assert row in README, (name, row)


def test_standard_market_numbers():
    a = ev.analyse(ev.Settings())
    assert (a.market.n, a.market.m, a.market.budgets.sum() > 0) == (6, 5, True)
    assert R.c(float(a.eq.p.sum()), 3) == R.c(float(a.market.budgets.sum()), 3) == "6,461" and int(np.argmax(a.eq.p)) + 1 == 5
    assert (R.sci(a.final("pr")), R.sci(a.final("tat")), R.sci(a.final("pg"))) == ("1,2e-15", "5,1e-03", "9,5e-10")
    assert (a.reached("pr", 1e-6), a.reached("tat", 1e-6), a.reached("pg", 1e-6)) == (601, None, 97)
    for part in ("**6,461**", "**1,2e-15**", "**5,1e-03**", "**9,5e-10**", "**601** Schritten", "**97** Funktionsauswertungen", "Slot 5"):
        assert part in README, part


def test_large_market_preset_numbers():
    preset = C.PRESETS["Großer Markt: das Endspiel"]
    a = ev.analyse(ev.Settings(**preset))
    assert (a.market.n, a.market.m, preset["steps"], preset["seed"]) == (30, 12, 10000, 17)
    assert (R.sci(a.final("pr")), R.sci(a.final("pg")), R.sci(a.final("tat"))) == ("1,7e-15", "2,8e-08", "3,3e-04")
    assert len(a.errs["pg"]) == C.PG_MAX_ITER == 3000
    for part in ("**1,7e-15**", "**2,8e-08**", "**3,3e-04**"):
        assert part in README, part


def test_the_tatonnement_fails_for_large_steps_on_the_small_market():
    counts = {}
    for c_eta in (0.25, 1.0, 2.0, 4.0, 8.0):
        n_bad = 0
        for seed in C.TABLE_SEEDS:
            mk = make_market("stetig", 6, 5, C.DEFAULT_BUDGETS, seed)
            eq = A.solve(mk.v, mk.budgets)
            errs, _ = A.run_tatonnement(mk.v, mk.budgets, R.ETA_STEPS, eq.p, c_eta)
            n_bad += errs[-1] > 1.0
        counts[c_eta] = n_bad
    assert counts == {0.25: 0, 1.0: 0, 2.0: 0, 4.0: 4, 8.0: 8}
    for part in ("bei $c = 4$ in **4 von 8** Märkten", "bei $c = 8$ in **8 von 8**", "bei $c \\le 2$ in keinem"):
        assert part in README, part
    preset = C.PRESETS["Tâtonnement zu großer Schritt"]
    a = ev.analyse(ev.Settings(**preset))
    assert (a.market.n, a.market.m, preset["eta"], preset["seed"], preset["steps"]) == (6, 5, 8.0, 17, 1500) and a.final("tat") > 1.0
    mk = a.market
    errs, _ = A.run_tatonnement(mk.v, mk.budgets, 3000, a.eq.p, preset["eta"])
    assert R.c(float(errs[300:].min()), 1) == "1,0" and R.c(float(errs[300:].max()), 1) == "6,6"
    for part in ("(6×5, Seed 17, $c = 8$, 1 500 Schritte) liegt er bei mehr als 1", "von Schritt 300 bis 3 000 zwischen 1,0 und 6,6"):
        assert part in README, part


def test_proportional_response_is_slow_in_one_of_the_eight_larger_markets():
    slow = []
    for seed in C.TABLE_SEEDS:
        mk = make_market("stetig", 20, 10, C.DEFAULT_BUDGETS, seed)
        eq = A.solve(mk.v, mk.budgets)
        errs, _ = A.run_pr(mk.v, mk.budgets, C.TABLE_PR_MAX, eq.p, stop_below=1e-6)
        if errs[-1] > 1e-6:
            slow.append((seed, R.sci(float(errs[-1]))))
    assert slow == [(200003, "6,0e-06")]
    assert "6,0e-06 (Seed 200003)" in README and "100 000 Schritten" in README


def test_the_gradient_reaches_1e_9_in_three_of_eight_larger_markets():
    """Tabelle 3: 38 % (3 von 8) bei 20×10; der Rest stößt an die Genauigkeitsgrenze der Liniensuche (Fehler bleibt zwischen 1e-9 und 1e-6)."""
    reached, floor = 0, []
    for seed in C.TABLE_SEEDS:
        mk = make_market("stetig", 20, 10, C.DEFAULT_BUDGETS, seed)
        eq = A.solve(mk.v, mk.budgets)
        errs, _, _ = A.run_gradient(mk.v, mk.budgets, C.TABLE_PG_MAX, eq.p)
        if errs.min() <= 1e-9:
            reached += 1
        else:
            floor.append(float(errs.min()))
    assert reached == 3 and max(floor) < 1e-6
    assert "erreicht aber 10⁻⁹ bei 20×10 nur in 38 % der Märkte" in README


def test_tie_numbers_of_the_readme():
    census = R.census_row()
    assert census == "| 0 von 30 | 28 von 30 | 0 von 30 |"
    for part in ("keiner einen Kreis im Träger, bei ganzzahligen Nutzen **28 von 30**", "bei dünn besetzten keiner", "**5 unabhängige Kreise**", "**3,4e-15**", "**0,12**"):
        assert part in README, part
    t = R.tie_numbers()
    assert t["rank"] == 5 and R.sci(t["price_diff"]) == "3,4e-15" and R.c(t["alloc_diff"]) == "0,12"


def test_lie_numbers_of_the_readme():
    n = R.lie_numbers()
    assert R.c(n["honest"], 3) == "3,000" and R.c(n["best"], 3) == "3,329" and R.c(n["ratio"]) == "2,93" and R.c(100 * (n["best"] / n["honest"] - 1), 1) == "11,0"
    assert len(C.MANIP_RATIOS) == 121
    for part in ("**3,000**, **3,250**, **3,329** bei Verhältnis 2,93 (**+11,0 %**)", "wahrer Nutzen **3,25**", "Raster von 121 Verhältnissen", "ist 2,93 mit Nutzen 3,329 (**+11,0 %**)"):
        assert part in README, part
    mk = ev.lie_market()
    assert A.true_utility_after_report(mk.v, mk.budgets, 0, np.array([1.0, 2.0])) == pytest.approx(3.25, abs=1e-12)


def test_search_numbers_of_the_readme():
    row = R.search_row()
    assert row == "| 8 von 20 | 0,0 % | 5,7 % | 23,8 % |" and row in README
    assert "in **8 von 20** Märkten" in README and "20 Zufallsmärkten (4 Spediteure, 3 Slots, 60 Meldungen je Markt)" in README
    assert (C.MANIP_MARKETS, C.MANIP_TRIES, C.MANIP_SIZE) == (20, 60, (4, 3))


def test_the_largest_misreport_gain_against_eisenberg_gale():
    so = pytest.importorskip("scipy.optimize")
    best = R.search_result()["best"]
    mk, report = best["market"], best["report"]
    vr = mk.v.copy(); vr[0] = report
    n, m = vr.shape

    def neg(z):
        return -(mk.budgets * np.log((vr * z.reshape(n, m)).sum(axis=1))).sum()

    cons = [{"type": "eq", "fun": (lambda z, j=j: z.reshape(n, m)[:, j].sum() - 1.0)} for j in range(m)]
    res = so.minimize(neg, np.full(n * m, 1.0 / n), constraints=cons, bounds=[(1e-9, 1.0)] * (n * m), method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    honest = float(mk.v[0] @ A.solve(mk.v, mk.budgets).x[0])
    gain = float(mk.v[0] @ res.x.reshape(n, m)[0]) / honest - 1.0
    assert R.c(100 * gain, 1) == "23,8" and gain == pytest.approx(best["gain"], abs=1e-6)
