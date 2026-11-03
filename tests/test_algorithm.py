"""Einheitentests des Kerns an von Hand gerechneten Mini-Märkten (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import fsh_algorithm as A
import fsh_constants as C


def test_hand_example_diagonal(diagonal):
    v, b = diagonal
    eq = A.solve(v, b)
    assert eq.exact and eq.p == pytest.approx([1.0, 1.0], abs=1e-12)
    assert eq.x == pytest.approx(np.eye(2), abs=1e-12) and A.utilities(v, eq.x) == pytest.approx([2.0, 2.0])


def test_hand_example_equal_values_has_unique_prices_but_not_a_unique_allocation(equal_values):
    v, b = equal_values
    eq = A.solve(v, b)
    assert eq.p == pytest.approx([2.0, 2.0], abs=1e-12)
    assert A.utilities(v, eq.x) == pytest.approx([0.5, 1.5], abs=1e-12)
    assert A.cycle_rank(A.tight_edges(v, eq.p)) == 1                                # ein Kreis: vier Kanten, vier Knoten, ein Zusammenhang
    alt = np.array([[0.5, 0.0], [0.5, 1.0]])                                        # eine andere Zuteilung mit denselben Preisen
    assert A.gaps(v, b, eq.p, alt)["bpb"] == pytest.approx(0.0, abs=1e-12) and A.gaps(v, b, eq.p, alt)["budget"] == pytest.approx(0.0, abs=1e-12)


def test_hand_example_indifferent_buyer(indifferent):
    v, b = indifferent
    eq = A.solve(v, b)
    assert eq.p == pytest.approx([1.0, 1.0], abs=1e-12) and A.utilities(v, eq.x) == pytest.approx([3.0, 1.0])


def test_hand_example_three_slots_with_a_tie(three_slots):
    v, b = three_slots
    eq = A.solve(v, b)
    assert eq.p == pytest.approx([1.0, 2.0 / 3.0, 4.0 / 3.0], abs=1e-12)
    assert eq.x == pytest.approx(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]), abs=1e-12)
    assert A.utilities(v, eq.x) == pytest.approx([4.0, 6.0])
    tight = A.tight_edges(v, eq.p)
    assert tight.tolist() == [[True, False, False], [False, True, True]]            # Käufer 2: 2/(2/3) = 4/(4/3) = 3


def test_cycle_rank_counts_independent_cycles():
    assert A.cycle_rank(np.eye(2, dtype=bool)) == 0
    assert A.cycle_rank(np.ones((2, 2), dtype=bool)) == 1
    assert A.cycle_rank(np.ones((3, 3), dtype=bool)) == 4                           # 9 Kanten - (6 Knoten - 1)
    assert A.cycle_rank(np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)) == 0          # Weg über vier Knoten


def test_price_error_is_the_largest_relative_deviation():
    assert A.price_error(np.array([1.1, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.1)
    assert A.price_error(np.array([1.0, 1.9]), np.array([1.0, 2.0])) == pytest.approx(0.05)


def test_projection_has_the_right_row_sums_and_respects_the_mask():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(5, 4))
    mask = rng.random((5, 4)) < 0.7
    mask[np.arange(5), rng.integers(0, 4, 5)] = True
    budgets = rng.random(5) + 0.5
    out = A.proj_rows(y, budgets, mask)
    assert out.sum(axis=1) == pytest.approx(budgets) and (out >= 0).all() and (out[~mask] == 0).all()


def test_projection_is_the_closest_feasible_point():
    """Kein zufälliger zulässiger Punkt liegt näher an y als die Projektion."""
    rng = np.random.default_rng(5)
    y = rng.normal(size=(3, 4))
    mask = np.ones((3, 4), dtype=bool)
    budgets = np.array([1.0, 2.0, 0.5])
    out = A.proj_rows(y, budgets, mask)
    best = np.linalg.norm(y - out, axis=1)
    for _ in range(300):
        z = rng.random((3, 4))
        z = z / z.sum(axis=1, keepdims=True) * budgets[:, None]
        assert (np.linalg.norm(y - z, axis=1) >= best - 1e-12).all()


def test_projection_of_a_point_that_is_already_feasible_is_the_point():
    y = np.array([[0.2, 0.3, 0.5]])
    assert A.proj_rows(y, np.array([1.0]), np.ones((1, 3), dtype=bool)) == pytest.approx(y)


def test_proportional_response_keeps_budgets_and_price_sum(diagonal):
    rng = np.random.default_rng(1)
    v, budgets = rng.random((5, 4)) + 0.05, rng.random(5) + 0.5
    b = A.initial_spend(v, budgets)
    for _ in range(20):
        b = A.pr_step(v, budgets, b)
        assert b.sum(axis=1) == pytest.approx(budgets) and b.sum() == pytest.approx(budgets.sum())


def test_the_equilibrium_is_a_fixed_point_of_proportional_response(three_slots):
    v, budgets = three_slots
    eq = A.solve(v, budgets)
    assert A.pr_step(v, budgets, eq.spend) == pytest.approx(eq.spend, abs=1e-12)


def test_tatonnement_keeps_the_price_sum():
    rng = np.random.default_rng(2)
    v, budgets = rng.random((6, 5)) + 0.05, rng.random(6) + 0.5
    _, p = A.run_tatonnement(v, budgets, 50, np.ones(5))
    assert p.sum() == pytest.approx(budgets.sum()) and (p > 0).all()


def test_gradient_never_increases_the_objective_and_stays_feasible():
    rng = np.random.default_rng(4)
    v, budgets = rng.random((6, 5)) + 0.05, rng.random(6) + 0.5
    logv = A.log_values(v)
    start = A.shmyrev(A.initial_spend(v, budgets), logv)
    eq = A.solve(v, budgets)
    values = []
    for iters in (1, 5, 25):
        _, _, b = A.run_gradient(v, budgets, iters, eq.p)
        values.append(A.shmyrev(b, logv))
        assert b.sum(axis=1) == pytest.approx(budgets) and (b >= 0).all()
    assert values[0] <= start + 1e-12 and values[1] <= values[0] + 1e-12 and values[2] <= values[1] + 1e-12
    assert values[2] >= A.shmyrev(eq.spend, logv) - 1e-12                            # nie unter dem Optimum


def test_run_functions_stop_early_and_return_shorter_arrays():
    rng = np.random.default_rng(6)
    v, budgets = rng.random((4, 3)) + 0.05, rng.random(4) + 0.5
    eq = A.solve(v, budgets)
    e_pr, _ = A.run_pr(v, budgets, 5000, eq.p, stop_below=1e-6)
    assert 0 < len(e_pr) < 5000 and e_pr[-1] <= 1e-6 and (e_pr[:-1] > 1e-6).all()
    e_pg, ev, _ = A.run_gradient(v, budgets, 500, eq.p, stop_below=1e-6)
    assert len(e_pg) == len(ev) < 500 and e_pg[-1] <= 1e-6
    assert (np.diff(ev) >= 1).all()                                                  # mindestens eine Auswertung je Schritt


def test_polish_recovers_the_exact_prices_from_a_rough_allocation(three_slots):
    v, budgets = three_slots
    _, _, rough = A.run_gradient(v, budgets, 60, np.ones(3))
    p, ok, spend = A.polish(v, budgets, rough)
    assert ok and p == pytest.approx([1.0, 2.0 / 3.0, 4.0 / 3.0], abs=1e-12) and spend.sum(axis=1) == pytest.approx(budgets)


def test_polish_refuses_a_wrong_support():
    v, budgets = np.array([[3.0, 1.0], [1.0, 3.0]]), np.array([1.0, 1.0])
    wrong = np.array([[0.0, 1.0], [1.0, 0.0]])                                       # Käufer kaufen das jeweils schlechtere Gut
    assert A.polish(v, budgets, wrong)[1] is False


def test_feasible_spend_has_the_right_margins_and_stays_on_tight_edges(equal_values, three_slots):
    v, budgets = equal_values
    spend, ok = A.feasible_spend(v, budgets, np.array([2.0, 2.0]))
    assert ok and spend.sum(axis=1) == pytest.approx(budgets) and spend.sum(axis=0) == pytest.approx([2.0, 2.0]) and (spend >= 0).all()
    v, budgets = three_slots
    p = np.array([1.0, 2.0 / 3.0, 4.0 / 3.0])
    spend, ok = A.feasible_spend(v, budgets, p)
    assert ok and spend == pytest.approx(np.array([[1.0, 0.0, 0.0], [0.0, 2.0 / 3.0, 4.0 / 3.0]]), abs=1e-12)     # hier ist die Zuteilung eindeutig
    assert (spend[~A.tight_edges(v, p)] == 0).all()


def test_feasible_spend_reports_wrong_prices():
    v, budgets = np.array([[3.0, 1.0], [1.0, 3.0]]), np.array([1.0, 1.0])
    assert A.feasible_spend(v, budgets, np.array([1.5, 0.5]))[1] is False          # Preise passen nicht zu den Budgets: keine zulässige Zuteilung


def test_gaps_detect_a_wrong_allocation(diagonal):
    v, budgets = diagonal
    eq = A.solve(v, budgets)
    swapped = eq.x[::-1]                                                              # jeder kauft das schlechtere Gut
    g = A.gaps(v, budgets, eq.p, swapped)
    assert g["bpb"] > 0.5 and g["envy"] > 0.5 and g["clearing"] == pytest.approx(0.0)
    half = eq.x * 0.5                                                                 # die Hälfte jedes Guts bleibt übrig
    assert A.gaps(v, budgets, eq.p, half)["clearing"] == pytest.approx(0.5)


def test_nash_beaten_counts_better_allocations(diagonal):
    v, budgets = diagonal
    eq = A.solve(v, budgets)
    assert A.nash_beaten(v, budgets, eq.x, np.random.default_rng(1)) == 0
    assert A.nash_beaten(v, budgets, eq.x[::-1], np.random.default_rng(1), trials=50) > 0


def _unique(v, budgets):
    eq = A.solve(v, budgets)
    tight = A.tight_edges(v, eq.p)
    return A.cycle_rank(tight), A.unique_allocation(tight, eq.spend, budgets), eq


def test_a_cycle_can_still_leave_only_one_allocation():
    """Randfall aus der unabhängigen Prüfung: Der Träger hat einen Kreis (Käufer 1 und 2, Slots 1 und 2), aber die Summen legen jede Ausgabe fest."""
    v = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    budgets = np.array([1.5, 1.0, 0.5])
    rank, unique, eq = _unique(v, budgets)
    assert eq.p == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    assert rank == 1 and unique
    assert eq.x == pytest.approx(np.array([[0.5, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]]), abs=1e-9)


def test_a_cycle_with_free_spending_means_several_allocations(equal_values):
    v, budgets = equal_values                                                        # 2x2 mit allen Kanten eng: ein Kreis, der sich verschieben lässt
    rank, unique, _ = _unique(v, budgets)
    assert rank == 1 and not unique
    v = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    rank, unique, _ = _unique(v, np.array([1.0, 2.0, 3.0]))
    assert rank == 4 and not unique


def test_a_forest_has_a_unique_allocation(diagonal, three_slots):
    for v, budgets in (diagonal, three_slots):
        rank, unique, _ = _unique(v, budgets)
        assert rank == 0 and unique


def test_unique_allocation_does_not_depend_on_which_feasible_allocation_is_given():
    """Ausgangspunkt eine Ecke (Ausgaben nur auf einem Teil der engen Kanten) oder eine Mitte: dasselbe Urteil, weil es um alle zulässigen Zuteilungen geht."""
    tight, budgets = np.ones((2, 2), dtype=bool), np.array([1.0, 1.0])
    assert not A.unique_allocation(tight, np.eye(2), budgets)
    assert not A.unique_allocation(tight, np.full((2, 2), 0.5), budgets)
    v = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    t3, b3 = A.tight_edges(v, np.ones(3)), np.array([1.5, 1.0, 0.5])
    assert A.unique_allocation(t3, np.array([[0.5, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]]), b3)
