# Fisher-Markt: Preise aus Budgets – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-fisher-markt-demo.streamlit.app/)** -->

Erstes Stück der Methodenreihe **Marktmechanismen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Ein Mechanismus macht aus Geboten, Budgets oder Überzeugungen Preise und eine Zuteilung; die Reihe prüft ihn an Effizienz, Anreizen, Budgetausgleich und Erlös und rechnet ihn als konvexes Programm. Vehikel: **ein Terminal teilt knappe Slots an Spediteure zu.** Die Spediteure haben Budgets und verschieden große Nutzen für jeden Slot; die Demo findet die Preise, bei denen jeder das Beste für sein Geld kauft und kein Slot übrig bleibt (das **Gleichgewicht des Fisher-Marktes**), und vergleicht drei Wege dorthin. Nachfolger sind die Positionsauktion (Stück 2) und das Pacing-Gleichgewicht (Stück 3), das beide zusammenführt.

## Warum dieses Problem

Ein Preis, der aus Budgets und Wünschen entsteht, ist der einfachste Mechanismus, den es gibt, und zugleich ein Lehrstück: Das Gleichgewicht ist die Lösung eines **konvexen Programms** (Eisenberg–Gale), seine Preise sind die Schattenpreise der Slots, und ganz verschiedene Verfahren finden es: ein Auktionator, der Preise nach der Nachfrage hebt, eine Dynamik, in der jeder sein Budget nach dem Nutzen umverteilt, und ein Abstiegsverfahren auf dem Programm. Die Kernfrage der Demo: **Wie schnell und wie genau findet jedes Verfahren das Gleichgewicht, und was sagt das Gleichgewicht über Anreize?**

## Modell

$n$ Spediteure mit Budgets $B_i$, $m$ Slots zu je einer Einheit, Nutzen $u_i(x_i) = \sum_j v_{ij} x_{ij}$ (linear). **Gleichgewicht:** Preise $p_j$ und Anteile $x_{ij}$, sodass jeder Slot ganz verkauft ist, jeder sein Budget ausgibt und nur Slots mit dem besten Nutzen je Preiseinheit ($v_{ij}/p_j$) gekauft werden. Die Preise sind eindeutig; die Zuteilung ist eindeutig, wenn der Graph der besten Kanten kreisfrei ist (bei einem Kreis meist nicht, in Entartungsfällen doch). Dasselbe als Optimierung: die Zuteilung maximiert $\sum_i B_i \log u_i$ (Eisenberg–Gale); mit den Ausgaben $b_{ij} = p_j x_{ij}$ als Variablen ist es das Shmyrev-Programm. Die Formeln stehen im 📐-Expander der App.

Die Märkte sind synthetisch (Zufallsnutzen, Zufallsbudgets, feste Seeds); es gibt keine Daten echter Spediteure oder Terminals.

## Methodik

Drei Verfahren laufen auf demselben Markt, ihr Fehler ist der relative Preisfehler $\max_j |p_j - p^*_j|/p^*_j$ gegen die Referenz $p^*$:

- **Proportional Response:** Jeder Spediteur verteilt sein Budget im nächsten Schritt im Verhältnis des Nutzenbeitrags jedes Slots im aktuellen Schritt. Keine Schrittweite, kein Preis wird gesetzt. Laut Literatur ist das gespiegeltes Gradientenverfahren auf dem Shmyrev-Programm.
- **Tâtonnement:** Ein Auktionator verändert die Preise mit der Überschussnachfrage, $p_j \leftarrow p_j \exp(\eta_t (D_j - 1))$, danach Normierung auf die Budgetsumme; die Schrittweite ist $\eta_t = c/(n\sqrt{t})$ mit einem Faktor $c$ (Standard 2).
- **Gradientenverfahren:** projizierter Abstieg auf dem Shmyrev-Programm mit Armijo-Rückverfolgung; gezählt werden **Funktionsauswertungen** (die Liniensuche rechnet mehrere je Schritt), höchstens 3 000 Schritte.

Die **Referenz** ist ein Gradientenlauf mit verdoppelter Schrittzahl, bis das **Polieren** gelingt: Aus dem Träger der genäherten Lösung (Kanten mit nennenswerter Ausgabe) wird ein lineares System für Preise und Ausgaben gelöst und auf allen Kanten geprüft. Damit sind die Preise auf Rechengenauigkeit eindeutig. Bei Gleichständen bestimmt ein Fluss auf den besten Kanten eine zulässige Zuteilung. Schritte und Auswertungen sind **keine Rechenzeit**; die Tabelle zählt verschiedene Dinge und benennt sie.

## Befunde (gemessen, keine Behauptungen)

Zufallsmärkte mit festen Seeds; Spediteure mit ungleichen Budgets; jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund |
|---|---|
| Standardmarkt (6 Spediteure, 5 Slots, Seed 17): Preissumme gegen Budgetsumme | beide **6,461**; der teuerste Slot ist Slot 5 |
| Wie genau sind die drei Verfahren nach 2 000 Schritten (Standardmarkt)? | Proportional Response **unter 1e-12**, Tâtonnement **5,1e-03**, Gradientenverfahren **unter 1e-7** |
| Wie schnell erreichen sie 10⁻⁶ (Standardmarkt)? | Proportional Response nach **601** Schritten, Gradientenverfahren nach **97** Funktionsauswertungen, Tâtonnement nicht |
| Großer Markt (30 Spediteure, 12 Slots, 10 000 Schritte) | Proportional Response **unter 1e-12**, Gradientenverfahren **zwischen 1e-9 und 1e-7** (Genauigkeitsgrenze der Liniensuche, je nach Markt und Rechner liegt sie irgendwo zwischen etwa 1e-8 und 1e-15), Tâtonnement **3,3e-04** |
| Handbeispiel zur Falschangabe: ehrlicher Nutzen, Nutzen bei der Meldung (1 : 2), beste Meldung | **3,000**, **3,250**, **3,329** bei Verhältnis 2,93 (**+11,0 %**) |

**Tabelle 1: Handbeispiele** (jeder Wert von Hand nachrechenbar). Spediteure in Zeilen.

| Beispiel | Nutzen | Budgets | Preise | Nutzen im Gleichgewicht |
|---|---|---|---|---|
| **A**: zwei Käufer, jeder will ein anderes Gut | (2, 1); (1, 2) | 1, 1 | 1,000; 1,000 | 2,000; 2,000 |
| **B**: gleiche Nutzen, Budgets 1 : 3 | (1, 1); (1, 1) | 1, 3 | 2,000; 2,000 | 0,500; 1,500 |
| **C**: ein Käufer ist gleichgültig | (3, 1); (1, 1) | 1, 1 | 1,000; 1,000 | 3,000; 1,000 |
| **D**: zwei Käufer, drei Güter, Gleichstand | (4, 2, 1); (1, 2, 4) | 1, 2 | 1,000; 0,667; 1,333 | 4,000; 6,000 |

**Tabelle 2: Gleichgewichtsprüfer** (20 Märkte je Zeile, Seeds 300000 bis 300019). Spalte „Abweichung“: größte Verletzung von Räumung, Budget, Bang-per-Buck und budgetgewichteter Neidfreiheit; „Nash“: Zufallszuteilungen (50 je Markt) mit höherer Nash-Wohlfahrt; „Kreis“: Märkte, in denen die engen Kanten (der Träger) einen Kreis enthalten (meist mehrere Zuteilungen).

| Nutzen | Größe | Märkte | Abweichung | Nash | Kreis |
|---|---|---|---|---|---|
| stetige Nutzen | 6×5 | 20 | unter 1e-9 | 0 | 0 von 20 |
| stetige Nutzen | 20×10 | 20 | unter 1e-9 | 0 | 0 von 20 |
| ganzzahlige Nutzen (Gleichstände) | 6×5 | 20 | unter 1e-9 | 0 | 20 von 20 |
| ganzzahlige Nutzen (Gleichstände) | 20×10 | 20 | unter 1e-9 | 0 | 20 von 20 |
| dünn besetzt (viele Nullen) | 6×5 | 20 | unter 1e-9 | 0 | 0 von 20 |
| dünn besetzt (viele Nullen) | 20×10 | 20 | unter 1e-9 | 0 | 0 von 20 |

**Tabelle 3: Drei Verfahren über acht Märkte** (stetige Nutzen, ungleiche Budgets, Seeds 200000 bis 200007). Median der Arbeit bis zum Preisfehler: Schritte bei Proportional Response und Tâtonnement, Funktionsauswertungen beim Gradientenverfahren. In Klammern der Anteil der Märkte, die das Ziel erreichen (fehlt die Klammer, alle). „Rauschgrenze“: nicht berichtet, siehe unten. Grenzen: Proportional Response 100 000 Schritte, Tâtonnement 20 000, Gradientenverfahren 3 000; Faktor $c = 2$.

| Markt | Preisfehler | Proportional Response | Tâtonnement | Gradientenverfahren |
|---|---|---|---|---|
| 6×5 | ≤ 1,0e-03 | 127 | 2037 | 26 |
| 6×5 | ≤ 1,0e-06 | 532 | nie | 41 |
| 6×5 | ≤ 1,0e-09 | 818 | nie | Rauschgrenze |
| 20×10 | ≤ 1,0e-03 | 422 | 965 | 89 |
| 20×10 | ≤ 1,0e-06 | 1620 (88 %) | nie | 180 |
| 20×10 | ≤ 1,0e-09 | 3015 (88 %) | nie | Rauschgrenze |

Das Gradientenverfahren braucht bis 10⁻⁶ die wenigsten Auswertungen (eine andere Einheit als die Schritte der übrigen Verfahren, keine Rechenzeit), stößt aber unterhalb von 10⁻⁷ an die Rauschgrenze seiner Liniensuche: Sie vergleicht Zielwerte, die sich dort nur noch im Rundungsrauschen unterscheiden. Dort hängt die Zahl der Märkte, die 10⁻⁹ erreichen, vom Rechner ab (3 von 8 auf Windows, 4 von 8 auf Linux bei 20×10); die Tabelle weist die Zelle deshalb als „Rauschgrenze“ aus. Das ist eine Eigenschaft dieser Umsetzung, nicht des Verfahrens. Proportional Response rechnet nur Quotienten und hat diese Grenze nicht; in einem der acht 20×10-Märkte bleibt sein Fehler aber auch nach 100 000 Schritten bei 6,0e-06 (Seed 200003), dort ist die Konvergenz langsam. Das Tâtonnement erreicht 10⁻³ und schwankt danach zwischen etwa 10⁻⁴ und 3·10⁻³, ohne 10⁻⁶ zu erreichen.

**Tabelle 4: Schrittweiten-Faktor des Tâtonnements.** Preisfehler nach 2 000 Schritten, Median über dieselben acht Märkte je Größe; verlieren mindestens vier Märkte die Preise ganz (Fehler über 100 %), steht deren Zahl.

| Faktor c | 6×5 | 20×10 |
|---|---|---|
| 0,25 | 1,9e-03 | 1,1e-02 |
| 1,00 | 2,4e-03 | 1,1e-03 |
| 2,00 | 3,6e-03 | 1,5e-03 |
| 4,00 | über 100 % in 4 von 8 | 2,6e-03 |
| 8,00 | über 100 % in 8 von 8 | 5,8e-03 |

Kleine Schritte sind langsam, große bringen in dieser Umsetzung den Auktionator aus dem Takt: Bei 6×5 bleibt der Preisfehler nach 2 000 Schritten über 100 %, bei $c = 4$ in **4 von 8** Märkten, bei $c = 8$ in **8 von 8**, bei $c \le 2$ in keinem. Bei 20×10 liegt der Median auch bei $c = 8$ nur bei 5,8e-03: Die Schwelle hängt von der Marktgröße ab, weil die Schrittweite $c/(n\sqrt{t})$ mit $n$ schrumpft. Im Preset „Tâtonnement zu großer Schritt“ (6×5, Seed 17, $c = 8$, 1 500 Schritte) liegt er bei mehr als 1 und springt von Schritt 300 bis 3 000 zwischen 1,0 und 6,6.

**Gleichstände.** Bei stetigen Nutzen hat in 30 Märkten (6×5) keiner einen Kreis im Träger, bei ganzzahligen Nutzen **28 von 30**, bei dünn besetzten keiner. Im Standardmarkt mit ganzzahligen Nutzen (Seed 17) hat der Träger **5 unabhängige Kreise**: Ein zweiter Lauf von Proportional Response aus zufälligem Start endet bei Preisen, die sich um **weniger als 1e-12** von der Referenz unterscheiden, aber bei einer Zuteilung, die sich um bis zu **0,12** je Eintrag unterscheidet; beide tragen dasselbe Gleichgewicht.

**Falschangabe.** Handbeispiel: Spediteur 1 hat die Nutzen (3, 1), Spediteur 2 die Nutzen (1, 3), Budgets 1 und 1. Ehrlich sind die Preise (1, 1) und der Nutzen 3. Meldet Spediteur 1 das Verhältnis 1 : 2, sind die Preise (2/3, 4/3) und er erhält Slot 1 ganz und ein Viertel von Slot 2: wahrer Nutzen **3,25** (von Hand gerechnet und gegen SLSQP auf dem Eisenberg–Gale-Programm geprüft). Die beste Meldung auf einem Raster von 121 Verhältnissen ist 2,93 mit Nutzen 3,329 (**+11,0 %**). In 20 Zufallsmärkten (4 Spediteure, 3 Slots, 60 Meldungen je Markt) findet Spediteur 1 eine gewinnbringende Falschangabe in **8 von 20** Märkten (mindestens so viele; die Suche ist nicht erschöpfend). Der Median von 0,0 % entsteht, weil die Märkte ohne Gewinn mitzählen:

| Märkte mit Gewinn | Gewinn im Median | 90-%-Quantil | Maximum |
|---|---|---|---|
| 8 von 20 | 0,0 % | 5,7 % | 23,8 % |

Der Markt ist also **nicht wahrheitsgetreu**. Das ist eine Suche über eine feste Familie von Meldungen (Streuung um den wahren Vektor oder Konzentration auf einen Teil der Slots), kein Beweis der besten Meldung; das Maximum von 23,8 % wird im Test gegen SLSQP auf dem Eisenberg–Gale-Programm nachgerechnet.

## Ehrliche Grenzen

- **Lineare Nutzen, teilbare Slots, eine Einheit je Slot:** Bei konkaven Nutzen (CES, Leontief) ändern sich Gleichgewicht und Verfahren; ganze Slots machen aus dem Markt ein Zuordnungsproblem.
- **Preisnehmer:** Der Markt nimmt die gemeldeten Nutzen als gegeben. Die Demo zeigt die Einzelabweichung eines Käufers, rechnet aber kein Nash-Gleichgewicht des Marktspiels (Brânzei u. a.).
- **Schritte und Auswertungen sind keine Rechenzeit:** Ein anderer Rechner, andere Schrittregeln oder Blockverfahren ändern die Reihenfolge; das Tâtonnement ist hier in einer einfachen gedämpften Form umgesetzt, kein Nachbau von Cole und Fleischer.
- **Genauigkeitsgrenze des Gradientenverfahrens:** gilt für diese Armijo-Liniensuche auf dem Zielwert; ein Verfahren mit Schrittweite aus der Gradientenlänge hätte sie nicht. Zahlen unterhalb von 10⁻⁷ sind plattformabhängig und werden nicht als Messung berichtet (die erste CI auf Linux hat das gezeigt).
- **Zufallsmärkte mit acht Wiederholungen** in Tabelle 3 und 4; die Mediane schwanken mit den Seeds. Größere Märkte als 30 Spediteure mit 12 Slots sind nicht Teil der Demo.
- **Literatur:** Alle Angaben beruhen auf Titel, Abstract und Sekundärquellen, nicht auf Volltexten; die Zuordnung „Proportional Response ist gespiegeltes Gradientenverfahren“ ist keiner einzelnen Arbeit zugeschrieben.

## Verifikation

- **Kern:** Gleichgewicht gegen Handrechnung (vier Handbeispiele), Projektion auf die Budgets (Eigenschaften und Nähe zum Zielpunkt), Proportional-Response-Schritt (Budget und Preissumme erhalten, Gleichgewicht ist Fixpunkt), Polieren und Fluss auf den besten Kanten (Randfälle und falsche Träger).
- **Orakel (anderer Rechenweg):** das Eisenberg–Gale-Programm in der Zuteilung, gelöst mit SLSQP (`scipy`): Preise auf allen Handbeispielen und in Zufallsmärkten aller drei Erzeugerarten (je sechs Seeds, drei Größen), Nash-Wohlfahrt gleich; ein langer Lauf Proportional Response gegen die Referenz; die Falschangabe im Handbeispiel von Hand und gegen SLSQP.
- **Eigenschaftstests:** alle Gleichgewichtsbedingungen in je drei Märkten pro Erzeugerart, Budgetart und Größe, Preissumme = Budgetsumme, Nash-Wohlfahrt nicht von Zufallszuteilungen übertroffen, Skalierung der Budgets und der Nutzenzeile eines Käufers, Umbenennen der Käufer und Slots, Monotonie im eigenen Budget, doppelter Slot doppelter Preis, budgetgewichtete Neidfreiheit.
- **Randfälle:** 2 Spediteure und 2 Slots für alle Erzeugerarten, gleiche Nutzen, mehr Slots als Käufer und umgekehrt, genau ein Wunsch, Reglergrenzen, Gleichstand an der Genauigkeitsgrenze, Worst-Case 30×12 mit 10 000 Schritten.
- **Verdikte** als reine Funktionen an handgesetzten Werten; **Presets, Layout, AppTest-Rauchtest** (jeder Button, jeder Regler an Minimum und Maximum), **Footer**, **Laufzeit-Importe**.
- **Behauptungen:** `tests/test_claims.py` rechnet jede Zahl dieser Datei nach.
- **Umfang:** 105 Tests, rund drei Minuten auf einem Windows-Rechner, gemessen (AppTest, Orakel mit scipy, Nachrechnen der README); `tools/readme_rows.py` ist kein Test, sondern erzeugt die Tabellen, die `test_claims.py` prüft.
- **Nicht gelaufen:** Mutationstests (seit 2026-10-28 nicht mehr Teil des Ablaufs); die Aussagenprüfung durch unabhängige Prüfer steht noch aus.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, drei Experimente, Grenzen, Formeln |
| `fsh_algorithm.py` | Kern: drei Verfahren, Polieren, Fluss, Prüfer, Falschangabe |
| `fsh_constants.py` | Regler-Grenzen, Seeds, Handbeispiele, Presets |
| `fsh_evaluation.py` | Analyse, Vergleich über viele Märkte, Gleichstände, Falschangaben, Verdikte (reine Funktionen) |
| `fsh_layout.py` | Fester Kopf (Stylesheet) |
| `fsh_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button, Auswahlregler |
| `fsh_scenario.py` | Märkte (stetig, ganzzahlig, dünn), Handbeispiele |
| `fsh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/readme_rows.py` | erzeugt die README-Tabellen über dieselben Funktionen wie die App |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für die SLSQP-Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Märkte, Orakel, Eigenschaften, Randfälle, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Andere Nutzenfunktionen** (CES, Leontief, quasilinear) und ein **Blockkoordinaten-Verfahren**: nur als Einordnung, nicht gerechnet.
- **Nash-Gleichgewichte des Marktspiels** und der Preis der Anarchie: Die Demo zeigt nur die Einzelabweichung eines Käufers.
- **Rechenzeit in Sekunden** und große Märkte mit Tausenden Teilnehmern.
- **Positionsauktion und Pacing** (Stücke 2 und 3 der Reihe), in denen aus dem Preis ein Auktionspreis wird.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; für die Streamlit Community Cloud ist die Python-Version im Dashboard einzustellen (3.12 empfohlen).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
