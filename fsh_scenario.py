"""Szenario: synthetische Fisher-Märkte (nur numpy). Zufall nur über den übergebenen Generator, nie über `hash()`.

Vehikel: Spediteure (Käufer) bieten um Slots eines Terminals (Güter, je eine Einheit). v[i, j] = Nutzen von Spediteur i je Einheit von Slot j (Zeitersparnis),
budgets[i] = Budget von Spediteur i. Keine Personen-, Firmen- oder Terminaldaten."""

from dataclasses import dataclass

import numpy as np

import fsh_constants as C


@dataclass(frozen=True, eq=False)
class Market:
    v: np.ndarray           # n x m Nutzen, jede Zeile hat einen positiven Eintrag, jede Spalte auch
    budgets: np.ndarray     # n Budgets > 0

    @property
    def n(self) -> int:
        return self.v.shape[0]

    @property
    def m(self) -> int:
        return self.v.shape[1]


def _repair(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jeder Käufer will mindestens ein Gut, jedes Gut hat mindestens einen Käufer (sonst ist der Preis nicht bestimmt)."""
    n, m = v.shape
    for i in range(n):
        if v[i].max() <= 0:
            v[i, rng.integers(m)] = rng.random() + 0.05
    for j in range(m):
        if v[:, j].max() <= 0:
            v[rng.integers(n), j] = rng.random() + 0.05
    return v


def random_values(kind: str, n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "stetig":
        return rng.random((n, m)) + 0.05
    if kind == "ganzzahlig":
        return rng.integers(1, 4, (n, m)).astype(float)
    if kind == "duenn":
        v = (rng.random((n, m)) + 0.05) * (rng.random((n, m)) < 0.5)
        return _repair(v, rng)
    raise ValueError(f"unbekannte Erzeugerart {kind!r}")


def random_budgets(kind: str, mode: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if mode == "gleich":
        return np.ones(n)
    if mode == "ungleich":
        return rng.integers(1, 3, n).astype(float) if kind == "ganzzahlig" else rng.random(n) + 0.5
    raise ValueError(f"unbekannte Budgetart {mode!r}")


def market_from_rng(kind: str, mode: str, n: int, m: int, rng: np.random.Generator) -> Market:
    v = random_values(kind, n, m, rng)
    return Market(v, random_budgets(kind, mode, n, rng))


def make_market(kind: str, n: int, m: int, mode: str, seed: int) -> Market:
    """Deterministisch aus (kind, n, m, mode, seed): gleiche Eingabe, gleicher Markt."""
    return market_from_rng(kind, mode, n, m, np.random.default_rng(seed))


def hand_market(key: str) -> Market:
    _, v, b = C.HAND[key]
    return Market(np.array(v, dtype=float), np.array(b, dtype=float))


def lie_market() -> Market:
    v, b = C.LIE_MARKET
    return Market(np.array(v, dtype=float), np.array(b, dtype=float))
