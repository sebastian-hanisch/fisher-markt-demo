"""Orakel-Tests des Kerns gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

Orakel: (1) das Eisenberg-Gale-Programm in der Zuteilung x, gelöst mit SLSQP (scipy; die Preise sind max_i B_i v_ij / u_i) - ein anderer Rechenweg als das Polieren auf dem
Träger, (2) die vier Handbeispiele von Hand, (3) ein langer Lauf Proportional Response gegen die Referenzpreise, (4) die Falschangabe im Handbeispiel (3,25 von Hand) und eine
Zufallsfalschangabe gegen SLSQP. Alle < 10 s."""

import numpy as np
import pytest

import fsh_algorithm as A
import fsh_constants as C
import fsh_scenario as S

so = pytest.importorskip("scipy.optimize")


def eisenberg_gale_prices(v, budgets):
    """Preise aus dem Eisenberg-Gale-Programm max sum_i B_i log u_i, sum_i x_ij = 1, x >= 0 (SLSQP): p_j = max_i B_i v_ij / u_i."""
    n, m = v.shape

    def neg(z):
        x = z.reshape(n, m)
        return -(budgets * np.log((v * x).sum(axis=1))).sum()

    cons = [{"type": "eq", "fun": (lambda z, j=j: z.reshape(n, m)[:, j].sum() - 1.0)} for j in range(m)]
    res = so.minimize(neg, np.full(n * m, 1.0 / n), constraints=cons, bounds=[(1e-9, 1.0)] * (n * m), method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    x = res.x.reshape(n, m)
    u = (v * x).sum(axis=1)
    return (budgets[:, None] * v / u[:, None]).max(axis=0), x


@pytest.mark.parametrize("key", sorted(C.HAND))
def test_hand_examples_against_eisenberg_gale(key):
    mk = S.hand_market(key)
    eq = A.solve(mk.v, mk.budgets)
    p_eg, _ = eisenberg_gale_prices(mk.v, mk.budgets)
    assert eq.p == pytest.approx(p_eg, rel=1e-6)


@pytest.mark.parametrize("kind", C.KINDS)
@pytest.mark.parametrize("size", [(3, 3), (4, 3), (6, 5)])
def test_prices_against_eisenberg_gale_on_random_markets(kind, size):
    for seed in range(6):
        mk = S.make_market(kind, size[0], size[1], "ungleich", seed)
        eq = A.solve(mk.v, mk.budgets)
        p_eg, _ = eisenberg_gale_prices(mk.v, mk.budgets)
        assert eq.p == pytest.approx(p_eg, rel=1e-5), (kind, size, seed)
        assert eq.exact


def test_the_nash_welfare_of_the_equilibrium_is_the_optimum_of_the_program():
    for seed in range(6):
        mk = S.make_market("stetig", 5, 4, "ungleich", seed)
        eq = A.solve(mk.v, mk.budgets)
        _, x_eg = eisenberg_gale_prices(mk.v, mk.budgets)
        assert A.nash_welfare(mk.v, mk.budgets, eq.x) == pytest.approx(A.nash_welfare(mk.v, mk.budgets, x_eg), abs=1e-7)
        assert A.nash_welfare(mk.v, mk.budgets, eq.x) >= A.nash_welfare(mk.v, mk.budgets, x_eg) - 1e-9


def test_a_long_proportional_response_run_ends_at_the_reference_prices():
    for seed in range(8):
        mk = S.make_market("stetig", 6, 5, "ungleich", seed)
        eq = A.solve(mk.v, mk.budgets)
        errs, b = A.run_pr(mk.v, mk.budgets, 30000, eq.p)
        assert errs[-1] < 1e-9, seed
        assert b.sum(axis=0) == pytest.approx(eq.p, rel=1e-8)


def test_the_tie_market_has_the_same_prices_but_another_allocation_in_a_second_run():
    mk = S.make_market("ganzzahlig", 6, 5, "ungleich", C.DEFAULT_SEED)
    eq = A.solve(mk.v, mk.budgets)
    assert A.cycle_rank(A.tight_edges(mk.v, eq.p)) > 0
    rng = np.random.default_rng(1)
    b0 = rng.random(mk.v.shape) * (mk.v > 0) + 1e-3
    b0 = b0 / b0.sum(axis=1, keepdims=True) * mk.budgets[:, None]
    _, b = A.run_pr(mk.v, mk.budgets, 20000, eq.p, b0)
    x2 = b / b.sum(axis=0)
    assert A.price_error(b.sum(axis=0), eq.p) < 1e-8
    assert np.abs(x2 - eq.x).max() > 1e-3                                        # andere Zuteilung ...
    g = A.gaps(mk.v, mk.budgets, eq.p, x2)
    assert max(g["clearing"], g["budget"], g["bpb"], g["envy"]) < 1e-7           # ... die ebenfalls ein Gleichgewicht trägt


def test_the_lie_example_by_hand(lie):
    v, b = lie
    honest = float(v[0] @ A.solve(v, b).x[0])
    assert honest == pytest.approx(3.0)
    # Meldung (1, 2): Preise (2/3, 4/3), Käufer 1 erhält Slot 1 ganz und ein Viertel von Slot 2: Nutzen 3 + 0,25 = 3,25
    vr = v.copy(); vr[0] = [1.0, 2.0]
    eq = A.solve(vr, b)
    assert eq.p == pytest.approx([2.0 / 3.0, 4.0 / 3.0], abs=1e-12) and eq.x[0] == pytest.approx([1.0, 0.25], abs=1e-12)
    assert A.true_utility_after_report(v, b, 0, np.array([1.0, 2.0])) == pytest.approx(3.25, abs=1e-12)
    # Meldung (2; 5,95) gegen SLSQP auf dem gemeldeten Markt
    vr[0] = [2.0, 5.95]
    _, x_eg = eisenberg_gale_prices(vr, b)
    assert A.true_utility_after_report(v, b, 0, np.array([2.0, 5.95])) == pytest.approx(float(v[0] @ x_eg[0]), abs=1e-6)


def test_random_reports_against_eisenberg_gale():
    rng = np.random.default_rng(9)
    mk = S.make_market("stetig", 4, 3, "ungleich", 4)
    for _ in range(12):
        report = mk.v[0] * np.exp(rng.normal(0.0, 0.7, mk.m))
        vr = mk.v.copy(); vr[0] = report
        _, x_eg = eisenberg_gale_prices(vr, mk.budgets)
        assert A.true_utility_after_report(mk.v, mk.budgets, 0, report) == pytest.approx(float(mk.v[0] @ x_eg[0]), abs=1e-6)
