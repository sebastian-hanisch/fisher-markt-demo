"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import fsh_evaluation as ev

OK_GAPS = {"clearing": 1e-15, "budget": 2e-15, "bpb": 3e-15, "envy": 1e-15}


def test_equilibrium_verdict_branches():
    kind, text = ev.verdict_equilibrium(OK_GAPS, True, 0, 6, 5)
    assert kind == "success" and "alle 5 Güter" in text and "alle 6 Budgets" in text and "3,0e-15" in text and "keine von 200 Zufallszuteilungen" in text
    kind, text = ev.verdict_equilibrium({**OK_GAPS, "bpb": 2e-9}, True, 0, 6, 5)                # über der Schwelle 1e-9
    assert kind == "error" and "2,0e-09" in text and "0 Zufallszuteilungen" in text
    kind, text = ev.verdict_equilibrium(OK_GAPS, True, 3, 6, 5)                                  # Gleichgewicht, aber eine Zufallszuteilung gewinnt
    assert kind == "error" and "3 Zufallszuteilungen" in text
    kind, text = ev.verdict_equilibrium({**OK_GAPS, "envy": -4e-9}, False, 0, 6, 5)              # nicht poliert: Warnung, Betrag der Abweichung
    assert kind == "warning" and "nicht auf Rechengenauigkeit poliert" in text and "4,0e-09" in text
    assert ev.verdict_equilibrium({**OK_GAPS, "bpb": 9.9e-10}, True, 0, 6, 5)[0] == "success"     # knapp unter der Schwelle


def test_methods_verdict_names_the_most_accurate_method_and_the_work():
    final = {"pr": 1.2e-15, "tat": 5.1e-3, "pg": 9.5e-10}
    kind, text = ev.verdict_methods(2000, final, {"pr": 601, "tat": None, "pg": 97})
    assert kind == "info" and "Nach 2000 Schritten (Gradientenverfahren: höchstens 3000)" in text
    for part in ("Proportional Response 1,2e-15", "Tâtonnement 5,1e-03", "Gradientenverfahren 9,5e-10", "Proportional Response erreicht 10⁻⁶ nach 601 Schritten",
                 "Tâtonnement erreicht 10⁻⁶ nicht", "Gradientenverfahren erreicht 10⁻⁶ nach 97 Funktionsauswertungen", "Am genauesten ist hier: **Proportional Response**"):
        assert part in text, part
    kind, text = ev.verdict_methods(500, {"pr": 1e-3, "tat": 1e-9, "pg": 1e-2}, {"pr": None, "tat": 400, "pg": None})
    assert kind == "success" and "**Tâtonnement**" in text and "Tâtonnement erreicht 10⁻⁶ nach 400 Schritten" in text      # Tâtonnement am besten und unter 1e-6
    assert ev.verdict_methods(500, {"pr": 1e-3, "tat": 5e-7, "pg": 1e-2}, {"pr": None, "tat": 450, "pg": None})[0] == "success"
    assert ev.verdict_methods(500, {"pr": 1e-3, "tat": 2e-6, "pg": 1e-2}, {"pr": None, "tat": None, "pg": None})[0] == "info"   # über der Schwelle 1e-6 bleibt es "info"


def test_ties_verdict_branches():
    kind, text = ev.verdict_ties(0, True, None, None)
    assert kind == "info" and "Kein Gleichstand" in text and "Wald" in text
    kind, text = ev.verdict_ties(1, False, 3.4e-15, 0.124)
    assert kind == "warning" and "einen Kreis" in text and "mehrere Zuteilungen" in text and "Preise sind trotzdem eindeutig" in text and "3,4e-15" in text and "0,12" in text
    assert "unendlich" not in text
    kind, text = ev.verdict_ties(5, False, 1e-12, 0.5)
    assert kind == "warning" and "5 unabhängige Kreise" in text and "1,0e-12" in text and "0,50" in text
    kind, text = ev.verdict_ties(1, True, 1e-15, 0.0)                                          # Kreis, aber jede Ausgabe festgelegt
    assert kind == "info" and "einen Kreis" in text and "trotzdem eindeutig" in text and "mehrere Zuteilungen" not in text


def test_lie_verdict_both_branches():
    kind, text = ev.verdict_lie(3.0, 3.329, 2.93, 1 / 3)
    assert kind == "warning" and "Verhältnis 0,33" in text and "Nutzen 3,000" in text and "Verhältnis 2,93" in text and "**3,329**" in text and "+11,0 %" in text
    kind, text = ev.verdict_lie(3.0, 3.0 + 1e-10, 0.33, 1 / 3)                                  # Rundungsrauschen zählt nicht als Gewinn
    assert kind == "success" and "lohnt keine Falschangabe" in text and "Nutzen 3,000" in text


def test_search_verdict_both_branches():
    kind, text = ev.verdict_search(8, 20, 0.0, 0.0568, 0.2377)
    assert kind == "warning" and "**8 von 20**" in text and "im Median 0,0 %" in text and "90-%-Quantil 5,7 %" in text and "Maximum 23,8 %" in text
    kind, text = ev.verdict_search(0, 20, 0.0, 0.0, 0.0)
    assert kind == "success" and "In keinem der 20 Märkte" in text


def test_steps_to_returns_the_work_at_the_first_hit():
    import numpy as np
    errs, work = np.array([1.0, 0.1, 0.01, 0.2]), np.array([10.0, 20.0, 30.0, 40.0])
    assert ev.steps_to(errs, work, 0.1) == 20 and ev.steps_to(errs, work, 0.01) == 30 and ev.steps_to(errs, work, 1e-3) is None
