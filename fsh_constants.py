"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import numpy as np

KINDS = ("stetig", "ganzzahlig", "duenn")
KIND_LABEL = {"stetig": "stetige Nutzen", "ganzzahlig": "ganzzahlige Nutzen (Gleichstände)", "duenn": "dünn besetzt (viele Nullen)"}
BUDGET_MODES = ("gleich", "ungleich")
BUDGET_LABEL = {"gleich": "alle gleich", "ungleich": "ungleich"}

# Regler. Komplexität am Maximum: Ein Gleichgewicht ist ein kleines konvexes Programm (n x m Variablen); die Verfahren rechnen je Schritt Matrix-Operationen
# der Größe n x m (n = 30, m = 12: Mikrosekunden bis Zehntel-Millisekunden). Das Gradientenverfahren läuft höchstens PG_MAX_ITER Schritte; ein Vergleich über viele
# Märkte rechnet TABLE_MARKETS Märkte (Worst-Case-Test in tests/test_edge_cases.py).
N_MIN, N_MAX, DEFAULT_N = 2, 30, 6
M_MIN, M_MAX, DEFAULT_M = 2, 12, 5
STEPS_MIN, STEPS_MAX, DEFAULT_STEPS, STEPS_STEP = 100, 10000, 2000, 100
ETA_MIN, ETA_MAX, DEFAULT_ETA, ETA_STEP = 0.25, 8.0, 2.0, 0.25      # Faktor c: Schrittweite des Tâtonnements eta_t = c / (n sqrt(t))
SEED_MAX, DEFAULT_SEED = 999999, 17
DEFAULT_KIND, DEFAULT_BUDGETS = "stetig", "ungleich"

PG_MAX_ITER = 3000          # Gradientenverfahren: höchstens so viele Schritte
POLISH_REL = 1e-5           # Polieren: eine Kante gehört zum Träger, wenn die Ausgabe mehr als diesen Anteil des Budgets beträgt
TIE_TOL = 1e-9              # zwei Güter sind gleich gut je Preis, wenn sie sich relativ um weniger unterscheiden
SOLVE_START, SOLVE_CAP = 100, 12800     # Gradientenverfahren der Referenzlösung: Schrittzahl wird verdoppelt, bis das Polieren gelingt
TARGETS = (1e-3, 1e-6, 1e-9)            # Preisfehler, bis zu denen die Verfahren verglichen werden

# Vergleich über viele Märkte (Knopf): feste Seeds, nie ein einzelner Seed (DEMO-PLAYBOOK §4)
TABLE_SEEDS = tuple(range(200000, 200008))      # 8 Märkte je Größe
TABLE_PR_MAX, TABLE_TAT_MAX, TABLE_PG_MAX = 100000, 20000, 3000
TABLE_SIZES = ((6, 5), (20, 10))
TIE_CENSUS_MARKETS, TIE_CENSUS_SIZE = 30, (6, 5)        # Zählung der Gleichstandsmärkte je Erzeugerart
MANIP_MARKETS, MANIP_TRIES, MANIP_SIZE = 20, 60, (4, 3)  # Suche nach Falschangaben in Zufallsmärkten
MANIP_GAIN_TOL = 1e-6
MANIP_RATIOS = tuple(float(x) for x in np.exp(np.linspace(np.log(0.1), np.log(10.0), 121)))   # gemeldetes Nutzenverhältnis Gut 2 : Gut 1
NASH_TRIALS = 200

COLOR_PR = "#2F6B65"
COLOR_TAT = "#c0392b"
COLOR_PG = "#4c78a8"
COLOR_TRUE = "#7a6aa8"
COLOR_GAIN = "#e08a2c"
COLOR_FAIL = "#c0392b"
METHOD_COLOR = {"pr": COLOR_PR, "tat": COLOR_TAT, "pg": COLOR_PG}
METHOD_LABEL = {"pr": "Proportional Response", "tat": "Tâtonnement", "pg": "Gradientenverfahren"}

# Handbeispiele: (Nutzenmatrix, Budgets) zum Nachrechnen von Hand
HAND = {
    "A": ("zwei Käufer, jeder will ein anderes Gut", ((2.0, 1.0), (1.0, 2.0)), (1.0, 1.0)),
    "B": ("gleiche Nutzen, Budgets 1 : 3", ((1.0, 1.0), (1.0, 1.0)), (1.0, 3.0)),
    "C": ("ein Käufer ist gleichgültig", ((3.0, 1.0), (1.0, 1.0)), (1.0, 1.0)),
    "D": ("zwei Käufer, drei Güter, Gleichstand", ((4.0, 2.0, 1.0), (1.0, 2.0, 4.0)), (1.0, 2.0)),
}
LIE_MARKET = (((3.0, 1.0), (1.0, 3.0)), (1.0, 1.0))   # Handbeispiel zur Falschangabe

PRESETS = {
    "Standardfall (Voreinstellung)": {"kind": "stetig", "n": DEFAULT_N, "m": DEFAULT_M, "budgets": DEFAULT_BUDGETS, "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "eta": DEFAULT_ETA},
    "Gleichstände: Preise eindeutig, Zuteilung nicht": {"kind": "ganzzahlig", "n": DEFAULT_N, "m": DEFAULT_M, "budgets": "ungleich", "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "eta": DEFAULT_ETA},
    "Großer Markt: das Endspiel": {"kind": "stetig", "n": 30, "m": 12, "budgets": DEFAULT_BUDGETS, "seed": DEFAULT_SEED, "steps": 10000, "eta": DEFAULT_ETA},
    "Tâtonnement zu großer Schritt": {"kind": "stetig", "n": DEFAULT_N, "m": DEFAULT_M, "budgets": DEFAULT_BUDGETS, "seed": DEFAULT_SEED, "steps": 1500, "eta": 8.0},
    "Dünner Markt": {"kind": "duenn", "n": 20, "m": 10, "budgets": DEFAULT_BUDGETS, "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "eta": DEFAULT_ETA},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Sechs Spediteure bieten um fünf Slots; Nutzen und Budgets sind Zufallszahlen. Alle drei Verfahren laufen auf demselben Markt.",
    "Gleichstände: Preise eindeutig, Zuteilung nicht": "Ganzzahlige Nutzen erzeugen Gleichstände: Mehrere Zuteilungen sind Gleichgewichte, die Preise sind trotzdem eindeutig.",
    "Großer Markt: das Endspiel": "30 Käufer, 12 Güter, 10 000 Schritte: Das Gradientenverfahren stößt an die Genauigkeitsgrenze seiner Liniensuche, das Tâtonnement bleibt weit darüber stehen.",
    "Tâtonnement zu großer Schritt": "Der Schrittweiten-Faktor 8 ist zu groß: Die Preise bleiben weit vom Gleichgewicht entfernt (Fehler über 100 %) und kommen nicht zur Ruhe.",
    "Dünner Markt": "Viele Käufer interessieren sich nur für wenige Güter: ein dünn besetzter Markt.",
}
