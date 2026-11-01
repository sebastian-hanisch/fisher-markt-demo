"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import time

import numpy as np
import pytest

import fsh_algorithm as A
import fsh_constants as C
import fsh_evaluation as ev
import fsh_scenario as S


def test_smallest_market_for_every_generator():
    for kind in C.KINDS:
        for mode in C.BUDGET_MODES:
            a = ev.analyse(ev.Settings(kind, C.N_MIN, C.M_MIN, mode, 3, C.STEPS_MIN, C.ETA_MIN))
            assert a.eq.exact and a.worst_gap < 1e-9, (kind, mode)
            assert all(np.isfinite(e).all() for e in a.errs.values())


def test_identical_values_and_budgets_split_the_budget_sum_evenly():
    v, b = np.ones((4, 8)), np.ones(4)
    eq = A.solve(v, b)
    assert eq.p == pytest.approx(np.full(8, 0.5), abs=1e-12)                    # Budgetsumme 4 auf 8 gleiche Slots
    assert A.utilities(v, eq.x) == pytest.approx(np.full(4, 2.0), abs=1e-12)


def test_more_slots_than_buyers_and_more_buyers_than_slots():
    for n, m in ((2, 12), (30, 2)):
        mk = S.make_market("stetig", n, m, "ungleich", 5)
        eq = A.solve(mk.v, mk.budgets)
        assert eq.exact and max(abs(x) for x in A.gaps(mk.v, mk.budgets, eq.p, eq.x).values()) < 1e-9, (n, m)


def test_a_buyer_who_wants_a_single_slot_spends_everything_on_it():
    v = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([1.0, 1.0])
    eq = A.solve(v, b)
    # Käufer 1 will nur Slot 1; Käufer 2 ist zwischen beiden gleichgültig: Slot 2 gehört ihm ganz, Slot 1 teilt er nicht mit Käufer 1 (der Preis steigt nur durch Käufer 1)
    assert eq.x[0] == pytest.approx([1.0, 0.0], abs=1e-12) and eq.p == pytest.approx([1.0, 1.0], abs=1e-12)


def test_the_generators_always_give_every_buyer_a_wish_and_every_slot_a_buyer():
    for kind in C.KINDS:
        for n, m in ((2, 2), (2, 12), (30, 2), (6, 5), (30, 12)):
            for seed in range(25):
                mk = S.make_market(kind, n, m, "ungleich", seed)
                assert (mk.v.max(axis=1) > 0).all() and (mk.v.max(axis=0) > 0).all() and (mk.budgets > 0).all(), (kind, n, m, seed)


def test_the_same_inputs_give_the_same_market():
    for kind in C.KINDS:
        a, b = S.make_market(kind, 6, 5, "ungleich", 9), S.make_market(kind, 6, 5, "ungleich", 9)
        assert np.array_equal(a.v, b.v) and np.array_equal(a.budgets, b.budgets)
        assert not np.array_equal(a.v, S.make_market(kind, 6, 5, "ungleich", 10).v)


def test_equal_budgets_are_all_one_and_unequal_budgets_are_not():
    assert S.make_market("stetig", 6, 5, "gleich", 1).budgets.tolist() == [1.0] * 6
    assert len(set(S.make_market("stetig", 6, 5, "ungleich", 1).budgets.tolist())) > 1
    assert set(S.make_market("ganzzahlig", 30, 5, "ungleich", 1).budgets.tolist()) <= {1.0, 2.0}


def test_the_largest_market_with_the_most_steps_runs_in_time():
    t = time.time()
    a = ev.analyse(ev.Settings("stetig", C.N_MAX, C.M_MAX, "ungleich", 5, C.STEPS_MAX, C.ETA_MAX))
    assert time.time() - t < 60 and a.eq.exact and all(np.isfinite(e).all() for e in a.errs.values())
    assert len(a.errs["pr"]) == C.STEPS_MAX and len(a.errs["pg"]) == C.PG_MAX_ITER


def test_the_gradient_runs_at_most_pg_max_iter_steps_even_for_few_steps():
    a = ev.analyse(ev.Settings(steps=C.STEPS_MIN))
    assert len(a.errs["pg"]) == C.STEPS_MIN and len(a.work["pg"]) == C.STEPS_MIN


def test_a_huge_tatonnement_step_stays_finite():
    a = ev.analyse(ev.Settings(eta=C.ETA_MAX))
    assert np.isfinite(a.errs["tat"]).all()


def test_the_equilibrium_of_a_market_with_one_tie_is_exact():
    """Ein Gleichstand an der Genauigkeitsgrenze (Nutzen gleich je Preis bis auf 1e-12) zählt als Gleichstand, nicht als Fehler."""
    v = np.array([[3.0, 3.0 * (1 + 1e-12)], [1.0, 1.0]])
    b = np.array([1.0, 1.0])
    eq = A.solve(v, b)
    assert eq.exact and eq.p.sum() == pytest.approx(2.0)
