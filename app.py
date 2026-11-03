"""Fisher-Markt: Preise aus Budgets - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 1 der Reihe „Marktmechanismen“: Spediteure mit Budgets bieten um knappe Slots eines Terminals. Die Preise, bei denen jeder das Beste für sein Geld kauft und kein Slot
übrig bleibt, sind das Gleichgewicht des Fisher-Marktes. Die Demo findet es mit drei Verfahren (Proportional Response, Tâtonnement, Gradientenverfahren), vergleicht sie und
zeigt, dass der Markt trotz eindeutiger Preise nicht wahrheitsgetreu ist.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import fsh_constants as C
from fsh_evaluation import (
    Settings,
    analyse,
    hand_rows,
    lie_curve,
    manipulation_search,
    methods_table,
    tie_census,
    tie_pair,
    verdict_equilibrium,
    verdict_lie,
    verdict_methods,
    verdict_search,
    verdict_ties,
)
from fsh_layout import dunkles_design, kopf_css
from fsh_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from fsh_scenario import make_market
from fsh_visualization import (
    build_convergence,
    build_lie_curve,
    build_spend,
    build_table_chart,
    build_tie_allocations,
)

import fsh_algorithm as A

st.set_page_config(page_title="Fisher-Markt – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Suche das Gleichgewicht und lasse die drei Verfahren laufen …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _hand():
    return hand_rows()


@st.cache_data(show_spinner=False)
def _tie(kind, n, m, budgets, seed):
    mk = make_market(kind, n, m, budgets, seed)
    eq = A.solve(mk.v, mk.budgets)
    return tie_pair(mk, eq, seed)


@st.cache_data(show_spinner=False)
def _census():
    return {kind: tie_census(kind) for kind in C.KINDS}


@st.cache_data(show_spinner=False)
def _lie():
    return lie_curve()


@st.cache_data(show_spinner="Suche Falschangaben in Zufallsmärkten …")
def _search():
    return manipulation_search()


@st.cache_data(show_spinner="Vergleiche die Verfahren über viele Märkte (das dauert etwa eine halbe Minute) …")
def _table():
    return methods_table()


def _de(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def _sci(v):
    return f"{v:.1e}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fsh_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🛒 Fisher-Markt: Preise aus Budgets")
st.markdown(
    """
Ein Terminal hat **knappe Slots**, Spediteure haben **Budgets** und verschieden große Nutzen für jeden Slot. Wenn jeder sein Geld ausgibt und nur kauft, was für ihn je Preiseinheit am meisten bringt, gibt es genau
einen Satz **Preise**, bei dem kein Slot übrig bleibt: das **Gleichgewicht des Fisher-Marktes**. Die Demo findet es mit **drei Verfahren** (Proportional Response, Tâtonnement, Gradientenverfahren), vergleicht,
wie schnell und wie genau sie sind, und zeigt zwei Eigenheiten: Bei **Gleichständen** sind die Preise eindeutig, die Zuteilung nicht, und der Markt ist **nicht wahrheitsgetreu** – wer falsche Nutzen meldet,
kann gewinnen. Erstes Stück der Reihe **Marktmechanismen**; alle Märkte sind synthetisch, es gibt keine Daten echter Spediteure oder Terminals.

Weiter unten: drei Abschnitte **🔬 Experiment**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Markt", expanded=True):
    st.markdown(
        """
1. **Der Markt.** $n$ Spediteure mit Budgets $B_i$, $m$ Slots zu je einer Einheit. Spediteur $i$ hat je Einheit von Slot $j$ den Nutzen $v_{ij}$; sein Nutzen ist die Summe über alle gekauften Anteile.
2. **Das Gleichgewicht.** Preise $p_j$ und Anteile $x_{ij}$, sodass (a) jeder Slot ganz verkauft ist, (b) jeder Spediteur sein Budget ausgibt und (c) jeder nur Slots mit dem **besten Nutzen je Preiseinheit** ($v_{ij}/p_j$) kauft.
3. **Dasselbe als Optimierung.** Die Zuteilung maximiert $\\sum_i B_i \\log u_i$ (Eisenberg–Gale); die Preise sind die Schattenpreise der Slots. Mit den Ausgaben $b_{ij} = p_j x_{ij}$ als Variablen ist es das Shmyrev-Programm.
4. **Drei Wege dorthin.** *Proportional Response:* jeder verteilt sein Budget im Verhältnis des Nutzenbeitrags. *Tâtonnement:* ein Auktionator hebt Preise bei Überschussnachfrage. *Gradientenverfahren:* abwärts auf dem Programm, projiziert auf die Budgets.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Markt**")
    kind = st.radio("Nutzen", C.KINDS, key="kind_radio", format_func=lambda k: C.KIND_LABEL[k], help="Wie die Nutzen erzeugt werden. Ganzzahlige Nutzen erzeugen Gleichstände.")
    budgets_mode = st.radio("Budgets", C.BUDGET_MODES, key="budgets_radio", format_func=lambda k: C.BUDGET_LABEL[k], help="Alle Spediteure gleich reich oder mit verschiedenen Budgets.")
    n = st.slider("Spediteure n", *bounds("n_slider"), key="n_slider", help="Wie viele Käufer es gibt.")
    m = st.slider("Slots m", *bounds("m_slider"), key="m_slider", help="Wie viele Güter es gibt (je eine Einheit).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt Nutzen und Budgets.")
    st.markdown("**Verfahren**")
    steps = st.slider("Schritte", *bounds("steps_slider"), key="steps_slider", step=C.STEPS_STEP, help="Wie viele Schritte Proportional Response und Tâtonnement laufen (das Gradientenverfahren höchstens 3000).")
    eta = st.slider("Schrittweiten-Faktor c des Tâtonnements", *bounds("eta_slider"), key="eta_slider", step=C.ETA_STEP,
                    help="Schrittweite des Tâtonnements c / (n √t). Klein: langsam, groß: das Verfahren kommt nicht zur Ruhe.")
    st.button("🎲 Neuer Markt", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_radio": kind, "budgets_radio": budgets_mode, "n_slider": int(n), "m_slider": int(m), "seed_input": int(seed), "steps_slider": int(steps), "eta_slider": round(float(eta), 2)})

settings = Settings(kind, int(n), int(m), budgets_mode, int(seed), int(steps), round(float(eta), 2))
a = _analysis(settings)
mk, eq = a.market, a.eq

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Preissumme", _de(float(eq.p.sum()), 3), delta=f"= Budgetsumme {_de(float(mk.budgets.sum()), 3)}", delta_color="off")
m2.metric("Teuerster Slot", f"Slot {int(np.argmax(eq.p)) + 1}", delta=f"Preis {_de(float(eq.p.max()), 3)}", delta_color="off")
m3.metric("Größte Abweichung", _sci(a.worst_gap), delta="Räumung, Budget, Bang-per-Buck, Neid", delta_color="off")
m4.metric("Kreise im Träger", f"{a.rank}", delta="Zuteilung eindeutig" if a.rank == 0 else "meist mehrere Zuteilungen", delta_color="off")
st.plotly_chart(build_spend(eq.spend, eq.p), width="stretch", key="main_chart")
st.caption("Ausgaben je Spediteur und Slot (Preis mal Anteil); über jeder Spalte steht der Preis des Slots. Jede Zeile summiert sich zum Budget des Spediteurs, jede Spalte zum Preis.")
if not eq.exact:
    st.warning("Das Polieren auf Rechengenauigkeit ist hier nicht gelungen; die Preise sind auf etwa 10⁻⁸ genau.")
with st.expander("Vier Handbeispiele zum Nachrechnen"):
    rows = ["| Beispiel | Nutzen (Zeilen: Spediteure) | Budgets | Preise | Nutzen im Gleichgewicht |", "|---|---|---|---|---|"]
    for r in _hand():
        v, b = C.HAND[r["key"]][1], C.HAND[r["key"]][2]
        vs = "; ".join("(" + ", ".join(f"{x:g}" for x in row).replace(".", ",") + ")" for row in v)
        rows.append(f"| **{r['key']}**: {r['name']} | {vs} | {', '.join(f'{x:g}' for x in b)} | {'; '.join(_de(float(x), 3) for x in r['p'])} | {'; '.join(_de(float(x), 3) for x in r['u'])} |")
    st.markdown("\n".join(rows))
    st.caption("Beispiel D: Spediteur 2 hat bei Slot 2 und Slot 3 denselben besten Nutzen je Preiseinheit (2 / 0,667 = 4 / 1,333 = 3) – ein Gleichstand.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist das ein Gleichgewicht – kauft jeder das Beste für sein Geld, und bleibt kein Slot übrig?")
kind_v, text_v = verdict_equilibrium(a.gaps, eq.exact, a.nash_beaten, mk.n, mk.m)
getattr(st, kind_v)(text_v)

st.markdown("---")

# --- Experiment 1: Drei Verfahren, ein Markt -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Drei Verfahren, ein Markt")
st.caption(f"Relativer Preisfehler max_j |p_j − p*_j| / p*_j über die Arbeit, auf dem Markt oben (p* = polierte Referenz; gepunktet: 10⁻³, 10⁻⁶, 10⁻⁹). "
           f"Proportional Response und Tâtonnement zählen Schritte, das Gradientenverfahren Funktionsauswertungen (Liniensuche) – verschiedene Dinge, deshalb getrennt benannt.")
st.plotly_chart(build_convergence(a.errs, a.work), width="stretch", key="convergence_chart")
final = {k: a.final(k) for k in ("pr", "tat", "pg")}
reached = {k: a.reached(k, 1e-6) for k in ("pr", "tat", "pg")}
kind_m, text_m = verdict_methods(settings.steps, final, reached)
getattr(st, kind_m)(text_m)
st.caption("Das Gradientenverfahren stößt je nach Markt und Rechner irgendwo zwischen etwa 10⁻⁸ und 10⁻¹⁵ an die Genauigkeitsgrenze seiner Liniensuche (sie vergleicht Zielwerte, die sich dort nur noch im Rundungsrauschen unterscheiden); "
           "das ist eine Eigenschaft dieser Umsetzung. Das Tâtonnement schwankt bei abklingender Schrittweite weit über 10⁻⁶; mit großem Faktor c kommt es in dieser Umsetzung nicht zur Ruhe.")
if st.button("📊 Vergleich über viele Märkte rechnen", key="table_button", help="Acht Zufallsmärkte je Größe, Arbeit bis 10⁻³, 10⁻⁶ und 10⁻⁹; dauert etwa eine halbe Minute."):
    st.session_state["table_done"] = True
if st.session_state.get("table_done"):
    table = _table()
    lines = ["| Markt | Preisfehler | Proportional Response (Schritte) | Tâtonnement (Schritte) | Gradientenverfahren (Auswertungen) |", "|---|---|---|---|---|"]
    for r in table:
        def cell(k):
            c = r[k]
            if c["share"] is None:
                return "Rauschgrenze"
            return "nie" if c["median"] is None else f"{c['median']}" + ("" if c["share"] == 1.0 else f" ({_de(100 * c['share'], 0)} %)")
        lines.append(f"| {r['n']}×{r['m']} | ≤ {_sci(r['target'])} | {cell('pr')} | {cell('tat')} | {cell('pg')} |")
    st.markdown("\n".join(lines))
    st.caption(f"Median über {len(C.TABLE_SEEDS)} Zufallsmärkte mit stetigen Nutzen und ungleichen Budgets (Seeds {C.TABLE_SEEDS[0]} bis {C.TABLE_SEEDS[-1]}); in Klammern der Anteil der Märkte, die das Ziel erreichen "
               f"(Grenzen: Proportional Response {C.TABLE_PR_MAX} Schritte, Tâtonnement {C.TABLE_TAT_MAX}, Gradientenverfahren {C.TABLE_PG_MAX}; Faktor c = {_de(C.DEFAULT_ETA, 1)}).")
    st.plotly_chart(build_table_chart(table, *C.TABLE_SIZES[-1]), width="stretch", key="table_chart")

st.markdown("---")

# --- Experiment 2: Gleichstände -----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Gleichstände – Preise eindeutig, Zuteilung nicht")
st.markdown("Hat ein Spediteur bei zwei Slots denselben besten Nutzen je Preiseinheit, darf er sein Geld beliebig auf beide verteilen. Zeichnet man die besten Kanten als Graph, bedeutet **ein Kreis** darin meist einen Freiheitsgrad der Zuteilung (in Entartungsfällen kann er festgelegt sein); "
            "ohne Kreis (ein Wald) ist die Zuteilung eindeutig. Die **Preise** sind trotzdem immer eindeutig.")
if a.rank == 0:
    kind_t, text_t = verdict_ties(0, None, None)
    getattr(st, kind_t)(text_t)
    st.caption("Der Markt oben hat keinen Gleichstand. Wählen Sie links „ganzzahlige Nutzen“ oder den Schnellstart „Gleichstände“, um zwei Zuteilungen zu sehen.")
else:
    x2, price_diff, alloc_diff = _tie(settings.kind, settings.n, settings.m, settings.budgets, settings.seed)
    st.plotly_chart(build_tie_allocations(eq.x, x2), width="stretch", key="tie_chart")
    kind_t, text_t = verdict_ties(a.rank, price_diff, alloc_diff)
    getattr(st, kind_t)(text_t)
    st.caption("Links die Zuteilung der Referenz, rechts die, bei der Proportional Response aus einem zufälligen Start landet (20 000 Schritte). Beide verkaufen jeden Slot ganz und lassen jedes Budget ausgeben.")
census = _census()
st.markdown(f"**Zählung** ({C.TIE_CENSUS_MARKETS} Märkte mit {C.TIE_CENSUS_SIZE[0]} Spediteuren und {C.TIE_CENSUS_SIZE[1]} Slots je Erzeugerart): Märkte mit mindestens einem Kreis im Träger – "
            + ", ".join(f"{C.KIND_LABEL[k]}: **{census[k]}**" for k in C.KINDS) + ".")

st.markdown("---")

# --- Experiment 3: Wer darf lügen? ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wer darf lügen? – der Markt ist nicht wahrheitsgetreu")
st.markdown("Der Markt nimmt die gemeldeten Nutzen als gegeben. Wer **falsche Nutzen** meldet, ändert die Preise – und kann damit mehr bekommen, als seine ehrliche Meldung ihm einbrächte. "
            "**Handbeispiel:** Zwei Spediteure, Budgets 1 und 1. Spediteur 1 hat die Nutzen (3, 1), Spediteur 2 die Nutzen (1, 3). Ehrlich kauft jeder sein Gut: Preise (1, 1), Nutzen 3. "
            "Meldet Spediteur 1 stattdessen das Verhältnis 1 : 2, sind beide Slots für ihn gleich gut je Preiseinheit; die Preise werden (2/3, 4/3), und er erhält Slot 1 ganz und ein Viertel von Slot 2: wahrer Nutzen 3,25.")
ratios = C.MANIP_RATIOS
us, honest, true_ratio = _lie()
st.plotly_chart(build_lie_curve(ratios, us, honest, true_ratio), width="stretch", key="lie_chart")
best_i = int(np.argmax(us))
kind_l, text_l = verdict_lie(honest, us[best_i], ratios[best_i], true_ratio)
getattr(st, kind_l)(text_l)
st.caption("Nur das Verhältnis der gemeldeten Nutzen zählt (Skalierung ändert das Gleichgewicht nicht). Gemeldet wird (1, r) mit r auf der waagerechten Achse.")
if st.button("🔎 In Zufallsmärkten nach Falschangaben suchen", key="search_button", help=f"{C.MANIP_MARKETS} Märkte mit {C.MANIP_SIZE[0]} Spediteuren und {C.MANIP_SIZE[1]} Slots, {C.MANIP_TRIES} Falschangaben je Markt."):
    st.session_state["search_done"] = True
if st.session_state.get("search_done"):
    res = _search()
    kind_s, text_s = verdict_search(res["wins"], res["markets"], res["median"], res["q90"], res["max"])
    getattr(st, kind_s)(text_s)
    st.caption(f"Spediteur 1 probiert {C.MANIP_TRIES} Meldungen je Markt (Streuung um den wahren Nutzenvektor oder Konzentration auf einen Teil der Slots) und behält die beste; "
               f"gemessen mit seinen wahren Nutzen. Das ist eine **Suche**, kein Beweis der besten Meldung; die Quote hängt von der Familie der Meldungen ab.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Lineare Nutzen** | Bei konkaven Nutzen (CES, Leontief) ändern sich Gleichgewicht und Verfahren; Proportional Response hat dafür eigene Konvergenzaussagen. | – |
| **Teilbare Slots, eine Einheit je Slot** | Ganze Slots (ganzzahlig) machen aus dem Markt ein Zuordnungsproblem, ein Gleichgewicht kann fehlen. | `auction-algorithm-demo` (ein Slot je Käufer) |
| **Käufer nehmen die Preise als gegeben** | Wer die Preise beeinflusst, kann gewinnen: Experiment 3. Als Spiel haben Fisher-Märkte mit linearen Nutzen reine Nash-Gleichgewichte (Brânzei u. a.); die Demo rechnet sie nicht. | Stück 2 (Positionsauktion: Anreize) |
| **Kein Geld im Nutzen** | Budgets sind Spielgeld, kein Nutzen; in Geldmärkten verschiebt sich das Gleichgewicht. | – |
| **Verfahrensvergleich auf Zufallsmärkten** | Schritte und Auswertungen sind keine Rechenzeit; ein anderer Rechner, andere Schrittregeln oder Blockverfahren ändern die Reihenfolge. | – |
| **Synthetische Märkte, keine Daten** | Echte Terminals haben Regeln, Fristen und Mindestmengen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gleichgewicht.** Käufer $i=1..n$ mit Budget $B_i$, Güter $j=1..m$ zu je einer Einheit, Nutzen $u_i(x_i) = \sum_j v_{ij}x_{ij}$. Preise $p \in \mathbb{R}^m_{>0}$ und Zuteilung $x$ bilden ein **Gleichgewicht**, wenn
$\sum_i x_{ij} = 1$ für alle $j$, $\sum_j p_j x_{ij} = B_i$ für alle $i$ und $x_{ij} > 0 \Rightarrow v_{ij}/p_j = \max_k v_{ik}/p_k$ (Bang-per-Buck). Es existiert; die Preise sind eindeutig; die Zuteilung ist eindeutig, wenn der Graph der besten Kanten kreisfrei ist (bei einem Kreis meist nicht, in Entartungsfällen doch).

**Eisenberg–Gale.** $\max \sum_i B_i \log u_i(x_i)$ unter $\sum_i x_{ij} \le 1$, $x \ge 0$. Die Optimalitätsbedingungen sind die Gleichgewichtsbedingungen, die Preise die Dualvariablen der Güterrestriktionen.

**Shmyrev.** Mit Ausgaben $b_{ij} = p_j x_{ij}$: $\min \sum_j p_j \log p_j - \sum_{ij} b_{ij}\log v_{ij}$ unter $p_j = \sum_i b_{ij}$, $\sum_j b_{ij} = B_i$, $b \ge 0$.

**Proportional Response.** $b_{ij}^{t+1} = B_i \cdot \dfrac{v_{ij}x_{ij}^t}{u_i(x_i^t)}$ mit $x_{ij}^t = b_{ij}^t/p_j^t$. Laut Literatur ist das gespiegeltes Gradientenverfahren (mirror descent) auf dem Shmyrev-Programm; für lineare Nutzen konvergiert es gegen das Gleichgewicht.

**Tâtonnement.** Nachfrage $D_j(p) = \sum_i s_{ij}/p_j$ mit Ausgaben $s_i$ auf den besten Gütern; $p_j \leftarrow p_j \exp(\eta_t(D_j - 1))$, danach Normierung auf $\sum_j p_j = \sum_i B_i$. In der Demo ist $\eta_t = c/(n\sqrt{t})$.

**Gradientenverfahren.** $b^{+} = \Pi(b - t\nabla f(b))$ mit $\nabla f(b)_{ij} = \log p_j + 1 - \log v_{ij}$ und der Projektion $\Pi$ auf $\{b\ge 0, \sum_j b_{ij} = B_i\}$ je Zeile; die Schrittweite $t$ wächst um den Faktor 1,5 und halbiert sich, bis $f(b^+) \le f(b) + \langle\nabla f(b), b^+-b\rangle + \|b^+-b\|^2/(2t)$.

**Polieren.** Aus dem Träger einer genäherten Lösung (Kanten mit nennenswerter Ausgabe): $p_j = \alpha_i v_{ij}$ auf den Kanten, Zeilen- und Spaltensummen der Ausgaben $B_i$ bzw. $p_j$: ein lineares System, das auf Rechengenauigkeit lösbar ist.

**Falschangabe.** Meldet Käufer $i$ $\hat v_i$ statt $v_i$, ist $(\hat p, \hat x)$ das Gleichgewicht des gemeldeten Marktes und sein wahrer Nutzen $v_i \cdot \hat x_i$; die Meldung lohnt, wenn er über dem ehrlichen liegt.

**Literatur** (nur Titel, Abstract und Sekundärquellen geprüft, keine Volltexte). Eisenberg, Gale: *Consensus of Subjective Probabilities: The Pari-Mutuel Method.* Annals of Mathematical Statistics 30(1), 1959.
Shmyrev: *An algorithm for finding equilibrium in the linear exchange model with fixed budgets.* Journal of Applied and Industrial Mathematics 3(4), 2009.
Zhang: *Proportional response dynamics in the Fisher market.* Theoretical Computer Science 2011.
Cole, Fleischer: *Fast-Converging Tatonnement Algorithms for One-Time and Ongoing Market Problems.* STOC 2008.
Gao, Kroer: *First-Order Methods for Large-Scale Market Equilibrium Computation.* NeurIPS 2020.
Brânzei, Chen, Deng, Filos-Ratsikas, Frederiksen, Zhang: *The Fisher Market Game: Equilibrium and Welfare.* AAAI 2014, S. 587–593.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Marktmechanismen: von Preisen bis Marktmachern](https://sebastianhanisch.net/konzepte-marktmechanismen.html)."
)
