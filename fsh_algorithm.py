"""Kern: Gleichgewicht eines Fisher-Marktes mit linearen Nutzen und drei Verfahren, es zu finden (nur numpy).

Markt: n Käufer mit Budgets B_i, m Güter zu je einer Einheit, Nutzen u_i(x_i) = sum_j v_ij x_ij. Gleichgewicht: Preise p_j, Zuteilung x_ij, jedes Gut ausverkauft,
jeder Käufer gibt sein Budget aus und kauft nur Güter mit dem besten Nutzen je Preiseinheit (Bang-per-Buck v_ij / p_j). Mit den Ausgaben b_ij = p_j x_ij ist das
das Shmyrev-Programm  min sum_j p_j log p_j - sum_ij b_ij log v_ij  mit p_j = sum_i b_ij, sum_j b_ij = B_i, b >= 0."""

from dataclasses import dataclass

import numpy as np

import fsh_constants as C


# --- Hilfsgrößen ----------------------------------------------------------------------------------------------------------------------------

def price_error(p: np.ndarray, ref: np.ndarray) -> float:
    """Größter relativer Preisfehler max_j |p_j - ref_j| / ref_j."""
    return float(np.max(np.abs(p - ref) / ref))


def initial_spend(v: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Startpunkt: jeder Käufer verteilt sein Budget gleichmäßig auf alle Güter, die er will."""
    b = (v > 0).astype(float)
    return b / b.sum(axis=1, keepdims=True) * budgets[:, None]


def log_values(v: np.ndarray) -> np.ndarray:
    mask = v > 0
    return np.where(mask, np.log(np.where(mask, v, 1.0)), 0.0)


def shmyrev(b: np.ndarray, logv: np.ndarray) -> float:
    p = b.sum(axis=0)
    if (p <= 0).any():
        return float("inf")
    return float((p * np.log(p)).sum() - (b * logv).sum())


def proj_rows(y: np.ndarray, budgets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Euklidische Projektion jeder Zeile auf {b >= 0, Summe = B_i, b = 0 außerhalb der Maske} (Sortierverfahren, zeilenweise vektorisiert)."""
    n, m = y.shape
    z = np.where(mask, y, -1e300)
    u = -np.sort(-z, axis=1)                                   # absteigend
    css = np.cumsum(u, axis=1) - budgets[:, None]
    k = np.arange(1, m + 1)[None, :]
    cond = u - css / k > 0
    rho = m - 1 - np.argmax(cond[:, ::-1], axis=1)             # letzter Index mit erfüllter Bedingung
    theta = css[np.arange(n), rho] / (rho + 1)
    return np.where(mask, np.maximum(y - theta[:, None], 0.0), 0.0)


def tight_edges(v: np.ndarray, p: np.ndarray, tol: float = C.TIE_TOL) -> np.ndarray:
    """Kanten (i, j), auf denen Käufer i den besten Nutzen je Preiseinheit hat (Gleichstände eingeschlossen)."""
    bpb = v / p
    return bpb >= bpb.max(axis=1, keepdims=True) * (1.0 - tol)


def cycle_rank(tight: np.ndarray) -> int:
    """Zahl unabhängiger Kreise des Käufer-Gut-Graphen der engen Kanten: Kanten - (Knoten - Zusammenhangskomponenten). 0 = Wald = Zuteilung eindeutig."""
    n, m = tight.shape
    parent = list(range(n + m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(tight)):
        parent[find(int(i))] = find(n + int(j))
    comps = len({find(a) for a in range(n + m)})
    return int(tight.sum()) - (n + m - comps)


def unique_allocation(tight: np.ndarray, spend: np.ndarray, budgets: np.ndarray) -> bool:
    """Gibt es zu den Preisen nur eine Zuteilung? `tight` = enge Kanten, `spend` = eine zulässige Zuteilung auf ihnen. Eine Kante kann genau dann in irgendeiner zulässigen
    Zuteilung positiv sein, wenn Hin- und Rückweg im Restgraphen (Käufer -> Gut auf allen engen Kanten, Gut -> Käufer auf Kanten mit Ausgabe) liegen, also beide Enden im selben
    starken Zusammenhang. Die Zuteilung ist eindeutig genau dann, wenn diese Kanten kreisfrei sind (ein Kreis lässt sich abwechselnd rauf- und runterschieben).
    Das ist die Bestimmung per LP (Minimum = Maximum jeder Ausgabe), nur ohne Löser."""
    n, m = tight.shape
    reach = np.eye(n + m, dtype=bool)
    reach[:n, n:] |= tight
    reach[n:, :n] |= (spend > 1e-9 * budgets.max()).T
    for k in range(n + m):                                     # transitiver Abschluss (Warshall)
        reach |= np.outer(reach[:, k], reach[k, :])
    movable = tight & reach[:n, n:] & reach[n:, :n].T
    return cycle_rank(movable) == 0


# --- Drei Verfahren -------------------------------------------------------------------------------------------------------------------------

def pr_step(v: np.ndarray, budgets: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ein Schritt Proportional Response: jeder Käufer verteilt sein Budget im Verhältnis des Nutzenbeitrags jedes Guts im aktuellen Schritt."""
    p = b.sum(axis=0)
    x = b / p
    u = (v * x).sum(axis=1)
    return budgets[:, None] * v * x / u[:, None]


def run_pr(v: np.ndarray, budgets: np.ndarray, steps: int, ref: np.ndarray, b0: np.ndarray | None = None, stop_below: float | None = None):
    """Proportional Response. Gibt (Preisfehler nach 1..steps Schritten, letzte Ausgaben) zurück; mit `stop_below` endet der Lauf, sobald der Fehler darunter liegt (kürzeres Feld)."""
    b = initial_spend(v, budgets) if b0 is None else b0.copy()
    errs = np.empty(steps)
    for t in range(steps):
        b = pr_step(v, budgets, b)
        errs[t] = price_error(b.sum(axis=0), ref)
        if stop_below is not None and errs[t] <= stop_below:
            return errs[:t + 1], b
    return errs, b


def run_tatonnement(v: np.ndarray, budgets: np.ndarray, steps: int, ref: np.ndarray, eta_factor: float = C.DEFAULT_ETA, stop_below: float | None = None):
    """Tâtonnement: Käufer geben ihr Budget für die besten Güter aus (Gleichstand gleich geteilt), Preise folgen der Überschussnachfrage
    p_j <- p_j exp(eta_t (D_j - 1)) mit eta_t = eta_factor / (n sqrt(t)), danach Normierung auf die Budgetsumme. Gibt (Preisfehler je Schritt, letzte Preise) zurück."""
    n, m = v.shape
    p = np.full(m, budgets.sum() / m)
    errs = np.empty(steps)
    for t in range(1, steps + 1):
        tight = tight_edges(v, p)
        spend = tight / tight.sum(axis=1, keepdims=True) * budgets[:, None]
        demand = spend.sum(axis=0) / p
        eta = eta_factor / (n * np.sqrt(t))
        p = p * np.exp(np.clip(eta * (demand - 1.0), -50.0, 50.0))
        p = p / p.sum() * budgets.sum()
        errs[t - 1] = price_error(p, ref)
        if stop_below is not None and errs[t - 1] <= stop_below:
            return errs[:t], p
    return errs, p


def run_gradient(v: np.ndarray, budgets: np.ndarray, iters: int, ref: np.ndarray, stop_below: float | None = None):
    """Projiziertes Gradientenverfahren auf dem Shmyrev-Programm mit Armijo-Rückverfolgung (Schrittweite wächst um 1,5 je Schritt und halbiert sich, bis der
    Abstieg stimmt). Gibt (Preisfehler je Schritt, Funktionsauswertungen insgesamt bis zu diesem Schritt, letzte Ausgaben) zurück."""
    mask = v > 0
    logv = log_values(v)
    b = initial_spend(v, budgets)
    t, f = 1.0, shmyrev(b, logv)
    errs, evals = np.empty(iters), np.empty(iters)
    total = 1
    for it in range(iters):
        grad = (np.log(b.sum(axis=0)) + 1.0)[None, :] - logv
        t = min(t * 1.5, 1e6)
        while True:
            cand = proj_rows(b - t * grad, budgets, mask)
            fc = shmyrev(cand, logv)
            total += 1
            d = cand - b
            if np.isfinite(fc) and fc <= f + float((grad * d).sum()) + float((d * d).sum()) / (2 * t) + 1e-15:
                break
            t *= 0.5
            if t < 1e-14:
                break
        b, f = cand, fc
        errs[it], evals[it] = price_error(b.sum(axis=0), ref), total
        if stop_below is not None and errs[it] <= stop_below:
            return errs[:it + 1], evals[:it + 1], b
    return errs, evals, b


# --- Das Gleichgewicht auf Rechengenauigkeit ----------------------------------------------------------------------------------------------

def polish(v: np.ndarray, budgets: np.ndarray, b: np.ndarray, rel: float = C.POLISH_REL):
    """Exakte Gleichgewichtspreise aus dem Träger einer genäherten Zuteilung. Auf den Kanten (i, j) mit b_ij > rel * B_i gilt p_j = alpha_i v_ij, die Ausgaben s_ij
    summieren sich zu B_i und zu p_j: ein lineares System in (s, p, alpha), gelöst mit lstsq. Gibt (p, Preise_ok, s) zurück; Preise_ok prüft Restfehler und
    Bang-per-Buck auf ALLEN Kanten (die Preise sind auch bei Gleichständen eindeutig, die Ausgaben s dann nicht)."""
    n, m = v.shape
    si, sj = np.nonzero(b > rel * budgets[:, None])
    k = len(si)
    size = k + m + n
    A = np.zeros((size, size))
    y = np.zeros(size)
    e = np.arange(k)
    A[e, k + sj] = 1.0                                   # p_j - alpha_i v_ij = 0
    A[e, k + m + si] = -v[si, sj]
    A[k + si, e] = 1.0                                   # sum_j s_ij = B_i
    y[k:k + n] = budgets
    A[k + n + sj, e] = 1.0                               # sum_i s_ij - p_j = 0
    A[k + n + np.arange(m), k + np.arange(m)] = -1.0
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    p, alpha = sol[k:k + m], sol[k + m:]
    ok = bool(
        (p > 0).all() and (alpha > 0).all() and np.abs(A @ sol - y).max() < 1e-10
        and ((v / p) <= (1.0 / alpha)[:, None] * (1.0 + 1e-9)).all()
    )
    spend = np.zeros((n, m))
    spend[si, sj] = sol[:k]
    return p, ok, spend


def feasible_spend(v: np.ndarray, budgets: np.ndarray, p: np.ndarray, tol: float = 1e-13):
    """Eine zulässige Zuteilung zu gegebenen Preisen als Fluss: Quelle -> Käufer (Kapazität B_i) -> Gut auf den engen Kanten (unbegrenzt) -> Senke (Kapazität p_j),
    gelöst mit kürzesten Erweiterungswegen (Edmonds-Karp). Bei Gleichständen gibt es viele Zuteilungen; der Fluss liefert eine davon.
    Gibt (Ausgaben, Zeilen- und Spaltensummen stimmen auf 1e-9) zurück."""
    n, m = v.shape
    tight = tight_edges(v, p)
    size = n + m + 2
    src, sink = 0, size - 1
    res = np.zeros((size, size))
    res[src, 1:n + 1] = budgets
    res[n + 1:n + m + 1, sink] = p
    res[1:n + 1, n + 1:n + m + 1] = np.where(tight, 1e9, 0.0)
    while True:
        parent = [-1] * size
        parent[src] = src
        queue = [src]
        for u in queue:
            for w in np.flatnonzero(res[u] > tol):
                if parent[w] < 0:
                    parent[w] = u
                    queue.append(int(w))
        if parent[sink] < 0:
            break
        path, w = [], sink
        while w != src:
            path.append((parent[w], w))
            w = parent[w]
        push = min(res[u, w] for u, w in path)
        for u, w in path:
            res[u, w] -= push
            res[w, u] += push
    spend = res[n + 1:n + m + 1, 1:n + 1].T.copy()              # Fluss = Rückkante Gut -> Käufer
    spend = np.where(tight, spend, 0.0)
    ok = max(np.abs(spend.sum(axis=1) - budgets).max(), np.abs(spend.sum(axis=0) - p).max()) < 1e-9
    return spend, bool(ok)


@dataclass(frozen=True, eq=False)
class Equilibrium:
    p: np.ndarray           # Preise
    spend: np.ndarray       # Ausgaben b_ij = p_j x_ij
    exact: bool             # Preise auf Rechengenauigkeit (Polieren gelungen)
    alloc_exact: bool       # Ausgaben erfüllen Zeilen- und Spaltensummen auf Rechengenauigkeit
    iters: int              # Gradientenschritte der Vorstufe

    @property
    def x(self) -> np.ndarray:
        return self.spend / self.p[None, :]


def solve(v: np.ndarray, budgets: np.ndarray, start: int = C.SOLVE_START, cap: int = C.SOLVE_CAP) -> Equilibrium:
    """Referenzlösung: Gradientenverfahren mit verdoppelter Schrittzahl, bis das Polieren auf dem Träger gelingt (Preise), dann Ausgaben aus dem Polieren
    (wenn nichtnegativ und die Summen stimmen) oder eine zulässige Zuteilung aus einem Fluss auf den engen Kanten. Gelingt das Polieren nie, bleiben die Preise des letzten Gradientenlaufs (exact = False)."""
    mask = v > 0
    iters = start
    while True:
        _, _, b = run_gradient(v, budgets, iters, np.ones(v.shape[1]))
        p, ok, spend = polish(v, budgets, b)
        if ok or iters >= cap:
            break
        iters *= 2
    if not ok:
        p = b.sum(axis=0)
        return Equilibrium(p, b, False, True, iters)
    margins = max(np.abs(spend.sum(axis=1) - budgets).max(), np.abs(spend.sum(axis=0) - p).max())
    if spend.min() >= -1e-12 and margins < 1e-10:
        return Equilibrium(p, np.maximum(spend, 0.0) * mask, True, True, iters)
    spend, flow_ok = feasible_spend(v, budgets, p)
    return Equilibrium(p, spend, True, flow_ok, iters)


# --- Prüfer und Eigenschaften -------------------------------------------------------------------------------------------------------------

def utilities(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (v * x).sum(axis=1)


def gaps(v: np.ndarray, budgets: np.ndarray, p: np.ndarray, x: np.ndarray) -> dict:
    """Verletzungen der Gleichgewichtsbedingungen (alle 0 im Gleichgewicht):
    clearing = Gut nicht ausverkauft, budget = Budget nicht ausgegeben, bpb = Käufer kauft nicht das Beste (best_i B_i / u_i - 1),
    envy = größte budgetgewichtete Missgunst max_{i,k} (v_i x_k / B_k - u_i / B_i)."""
    u = utilities(v, x)
    best = (v / p).max(axis=1)
    envy = (v @ x.T) / budgets[None, :] - (u / budgets)[:, None]     # [i, k] = v_i x_k / B_k - u_i / B_i
    return {
        "clearing": float(np.abs(x.sum(axis=0) - 1.0).max()),
        "budget": float(np.abs((x * p[None, :]).sum(axis=1) - budgets).max()),
        "bpb": float(np.max(best * budgets / u - 1.0)),
        "envy": float(envy.max()),
    }


def nash_welfare(v: np.ndarray, budgets: np.ndarray, x: np.ndarray) -> float:
    """Budgetgewichtete Nash-Wohlfahrt sum_i B_i log u_i (das Eisenberg-Gale-Ziel)."""
    return float((budgets * np.log(utilities(v, x))).sum())


def nash_beaten(v: np.ndarray, budgets: np.ndarray, x: np.ndarray, rng: np.random.Generator, trials: int = C.NASH_TRIALS, tol: float = 1e-9) -> int:
    """Zahl zufälliger zulässiger Zuteilungen (jedes Gut verteilt sich vollständig), die die Nash-Wohlfahrt des Gleichgewichts übertreffen (0 im Gleichgewicht)."""
    n, m = v.shape
    base = nash_welfare(v, budgets, x)
    beaten = 0
    for _ in range(trials):
        y = rng.random((n, m)) + 1e-12
        y /= y.sum(axis=0)
        beaten += nash_welfare(v, budgets, y) > base + tol
    return beaten


def true_utility_after_report(v: np.ndarray, budgets: np.ndarray, who: int, report: np.ndarray) -> float:
    """Wahrer Nutzen von Käufer `who`, wenn er `report` statt seiner Nutzenzeile meldet (Gleichgewicht des gemeldeten Marktes); NaN, wenn dessen Preise nicht poliert werden konnten."""
    vr = v.copy()
    vr[who] = report
    eq = solve(vr, budgets, start=50)
    return float(v[who] @ eq.x[who]) if eq.exact else float("nan")
