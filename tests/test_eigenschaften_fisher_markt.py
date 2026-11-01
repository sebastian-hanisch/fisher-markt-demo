"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jeden Markt gelten müssen, unabhängig von einem Vergleichswert."""

import numpy as np
import pytest

import fsh_algorithm as A
import fsh_constants as C
import fsh_scenario as S

CASES = [(kind, mode, size) for kind in C.KINDS for mode in C.BUDGET_MODES for size in ((3, 3), (6, 5))]


def _markets(count=3):
    for kind, mode, (n, m) in CASES:
        for seed in range(count):
            yield kind, S.make_market(kind, n, m, mode, 100 + seed)


def test_every_equilibrium_condition_holds_in_every_market():
    for kind, mk in _markets():
        eq = A.solve(mk.v, mk.budgets)
        g = A.gaps(mk.v, mk.budgets, eq.p, eq.x)
        assert eq.exact and max(abs(x) for x in g.values()) < 1e-9, (kind, g)
        assert (eq.x >= -1e-12).all() and (eq.p > 0).all()


def test_the_price_sum_is_the_budget_sum():
    for kind, mk in _markets():
        assert A.solve(mk.v, mk.budgets).p.sum() == pytest.approx(mk.budgets.sum(), rel=1e-10), kind


def test_nash_welfare_is_not_beaten_by_random_allocations():
    for kind, mk in _markets(2):
        eq = A.solve(mk.v, mk.budgets)
        assert A.nash_beaten(mk.v, mk.budgets, eq.x, np.random.default_rng(1), trials=100) == 0, kind


def test_scaling_all_budgets_scales_the_prices_and_leaves_the_allocation():
    for kind, mk in _markets(2):
        if kind == "ganzzahlig":
            continue                                                  # bei Gleichständen ist die Zuteilung nicht eindeutig
        e1, e2 = A.solve(mk.v, mk.budgets), A.solve(mk.v, 2.5 * mk.budgets)
        assert e2.p == pytest.approx(2.5 * e1.p, rel=1e-8) and e2.x == pytest.approx(e1.x, abs=1e-8), kind


def test_scaling_one_buyers_values_changes_nothing():
    for kind, mk in _markets(2):
        if kind == "ganzzahlig":
            continue
        v2 = mk.v.copy(); v2[0] *= 7.0
        e1, e2 = A.solve(mk.v, mk.budgets), A.solve(v2, mk.budgets)
        assert e2.p == pytest.approx(e1.p, rel=1e-8) and e2.x == pytest.approx(e1.x, abs=1e-8), kind


def test_renaming_the_buyers_only_permutes_the_allocation():
    rng = np.random.default_rng(2)
    for kind, mk in _markets(2):
        if kind == "ganzzahlig":
            continue
        perm = rng.permutation(mk.n)
        e1, e2 = A.solve(mk.v, mk.budgets), A.solve(mk.v[perm], mk.budgets[perm])
        assert e2.p == pytest.approx(e1.p, rel=1e-8) and e2.x == pytest.approx(e1.x[perm], abs=1e-8), kind


def test_renaming_the_slots_only_permutes_the_prices():
    rng = np.random.default_rng(3)
    for kind, mk in _markets(2):
        if kind == "ganzzahlig":
            continue
        perm = rng.permutation(mk.m)
        e1, e2 = A.solve(mk.v, mk.budgets), A.solve(mk.v[:, perm], mk.budgets)
        assert e2.p == pytest.approx(e1.p[perm], rel=1e-8), kind


def test_a_larger_own_budget_never_lowers_the_own_utility():
    for kind, mk in _markets(2):
        if kind == "ganzzahlig":
            continue
        base = A.utilities(mk.v, A.solve(mk.v, mk.budgets).x)
        for i in range(mk.n):
            b = mk.budgets.copy(); b[i] *= 1.5
            assert A.utilities(mk.v, A.solve(mk.v, b).x)[i] >= base[i] - 1e-9, (kind, i)


def test_a_slot_that_is_twice_as_valuable_to_everyone_costs_twice_as_much():
    """Ein zweiter Slot mit durchweg doppelten Nutzen ist derselbe Slot in doppelter Menge: Preis doppelt."""
    for seed in range(6):
        mk = S.make_market("stetig", 4, 3, "ungleich", seed)
        v2 = np.hstack([mk.v, 2.0 * mk.v[:, :1]])
        eq = A.solve(v2, mk.budgets)
        assert eq.p[3] == pytest.approx(2.0 * eq.p[0], rel=1e-8), seed


def test_the_price_of_a_slot_nobody_prefers_is_driven_by_the_buyers_who_do_want_it():
    """Ein Slot, den nur ein Käufer will, kostet genau soviel, wie dieser dafür ausgibt; der Käufer gibt sein ganzes Budget für ihn aus, wenn er sonst nichts will."""
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0]])
    b = np.array([0.7, 1.0, 1.5])
    eq = A.solve(v, b)
    assert eq.p[0] == pytest.approx(0.7, abs=1e-12)


def test_unequal_budgets_give_the_richer_buyer_at_least_as_much_utility_when_values_are_equal():
    v = np.ones((3, 4))
    b = np.array([1.0, 2.0, 3.0])
    u = A.utilities(v, A.solve(v, b).x)
    assert u == pytest.approx(b / b.sum() * 4.0, abs=1e-12)
    assert (np.diff(u) > 0).all()


def test_envy_freeness_is_budget_weighted_not_absolute():
    """Mit ungleichen Budgets beneidet der Ärmere den Reicheren um dessen Bündel (absolut), aber nicht nach Budget gewichtet."""
    v = np.ones((2, 3))
    b = np.array([1.0, 3.0])
    eq = A.solve(v, b)
    u = A.utilities(v, eq.x)
    assert (v[0] @ eq.x[1]) > u[0] + 1e-9                                       # absolut beneidet der Ärmere
    assert A.gaps(v, b, eq.p, eq.x)["envy"] < 1e-12                              # gewichtet nicht


def test_proportional_response_never_leaves_the_budget_simplex():
    for kind, mk in _markets(2):
        b = A.initial_spend(mk.v, mk.budgets)
        for _ in range(30):
            b = A.pr_step(mk.v, mk.budgets, b)
            assert b.sum(axis=1) == pytest.approx(mk.budgets) and (b >= 0).all() and (b[mk.v == 0] == 0).all(), kind
