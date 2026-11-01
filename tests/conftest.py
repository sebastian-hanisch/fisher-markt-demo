"""Gemeinsame Fixtures: die Mini-Märkte, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def diagonal():
    """Zwei Käufer, jeder will sein Gut (v = ((2, 1), (1, 2)), Budgets 1, 1): Preise (1, 1), jeder kauft sein Gut, Nutzen (2, 2)."""
    return np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 1.0])


@pytest.fixture
def equal_values():
    """Gleiche Nutzen, Budgets 1 und 3: Preise (2, 2), Nutzen (0,5; 1,5). Die Zuteilung ist nicht eindeutig (Gleichstand), die Preise schon."""
    return np.ones((2, 2)), np.array([1.0, 3.0])


@pytest.fixture
def indifferent():
    """v = ((3, 1), (1, 1)), Budgets 1, 1: Preise (1, 1), Käufer 1 kauft Slot 1, Käufer 2 Slot 2, Nutzen (3, 1)."""
    return np.array([[3.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0])


@pytest.fixture
def three_slots():
    """v = ((4, 2, 1), (1, 2, 4)), Budgets 1, 2: Preise (1; 2/3; 4/3), Käufer 1 kauft Slot 1 ganz, Käufer 2 Slot 2 und 3 ganz (Gleichstand), Nutzen (4, 6)."""
    return np.array([[4.0, 2.0, 1.0], [1.0, 2.0, 4.0]]), np.array([1.0, 2.0])


@pytest.fixture
def lie():
    """v = ((3, 1), (1, 3)), Budgets 1, 1: ehrlich Preise (1, 1) und Nutzen (3, 3); meldet Käufer 1 das Verhältnis (1, 2), sind die Preise (2/3, 4/3) und sein wahrer Nutzen 3,25."""
    return np.array([[3.0, 1.0], [1.0, 3.0]]), np.array([1.0, 1.0])
