# Sekretärproblem und Prophet – Angebote einholen, ohne zurück zu können – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-sekretaerproblem-demo.streamlit.app/)** -->

Fünftes Stück der Reihe **Online-Algorithmen** („Entscheiden ohne Kenntnis der Zukunft“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, unabhängiger Ast neben [paging-demo](https://github.com/sebastian-hanisch/paging-demo), [online-packen-demo](https://github.com/sebastian-hanisch/online-packen-demo) und [online-allokation-demo](https://github.com/sebastian-hanisch/online-allokation-demo) unter der Wurzel [ski-rental-demo](https://github.com/sebastian-hanisch/ski-rental-demo).
Anders als die Fall-Demos (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Konzept, **eine einzige Wahl ohne Zurück**, an einem wachsenden Beispiel: erst wenn man nur die **Rangfolge** der bisherigen Angebote kennt (Sekretärproblem), dann wenn man die **Verteilungen** kennt (Prophet-Ungleichung).
Kante zum Vorgänger: Online-Allokation entscheidet über **viele** Buchungen mit Preisen für knappen Platz; hier wird **ein einziges** Angebot gewählt, und die Frage ist, **was man dabei über die Angebote weiß**.

**Verwandte bestehende Demos:** `uld-buchungsannahme-demo` (Annahme mit Gewichtslimit und bekannter Verteilung, exakte dynamische Programmierung), `routenresilienz-demo` (Entscheiden unter Unsicherheit ohne Wiederholung), `online-matching-demo` (Zuordnen ohne Kenntnis der Zukunft).

## Warum dieses Problem

Angebote kommen nacheinander; jedes wird sofort genommen oder für immer verworfen, und gewählt wird genau eines. Das passt zur Wohnungssuche, zum Gebrauchtwagenkauf, zur Einstellung. Die Kernfrage: **Wie viel kann man herausholen, und woran hängt es, was man über die Angebote weiß?** Zwei Antworten stehen nebeneinander: Mit bloßer Rangfolge und zufälliger Reihenfolge findet man den Besten in etwa **37 %** der Fälle. Mit bekannten Verteilungen sichert eine einzige Schwelle die **Hälfte** dessen, was ein Prophet erreichen würde, auch bei einer Reihenfolge, die ein Gegner wählt.

## Modell

**Teil A (Sekretärproblem).** n Angebote mit verschiedenen Werten, gleichverteilt in (0, 1), kommen in zufälliger Reihenfolge; nach jedem ist nur bekannt, ob es das beste bisher ist. **Regel:** die ersten k nur ansehen, dann das erste Angebot nehmen, das besser ist als alle bisherigen; kommt keines, das letzte. Bei zufälliger Reihenfolge ist die Erfolgswahrscheinlichkeit exakt P(n, k) = (k/n) · Σ 1/i für i = k bis n − 1 (für k = 0: 1/n). Ein Regler legt fest, welcher Anteil der Folge am Anfang **aufsteigend sortiert** ist: Damit lässt sich die Annahme der zufälligen Reihenfolge verletzen.
**Teil B (Prophet-Ungleichung).** Das scharfe Beispiel: ein sicheres Angebot mit Wert 1 und ein riskantes mit Wert 1/ε und Wahrscheinlichkeit ε (sonst 0). Der Prophet sieht beide Werte und nimmt das größere, im Mittel E[max] = 2 − ε. Vier Regeln: **Schwelle E[max]/2**, **Schwelle am Median des Maximums**, **immer das erste** und **optimal** (Rückwärtsrechnung, kennt die Reihenfolge). Alle Zahlen in Teil B sind exakt gerechnet, nicht geschätzt. Formeln im 📐-Expander der App.

## Methodik

| Abschnitt der App | Was gerechnet wird |
|---|---|
| Teil A, Ergebnis | gewählte Einstellungen: 4000 gezogene Folgen (fester Seed), dazu die exakte Formel; eine gezeichnete Folge; Erfolgskurve über k |
| Experiment 1 | n = 100, k = 37, den Anfang der Folge zu 0 / 25 / 50 / 75 / 100 % aufsteigend sortiert, dazu absteigend; je 50 000 gezogene Folgen |
| Experiment 2 | n = 100, zufällig, k = 0 bis 70; Erfolg, Wert/Maximum und Anteil unter den besten 5; je 50 000 gezogene Folgen |
| Experiment 3 | exakt: bestes k und sein Erfolg für n = 3 bis 10 000 |
| Teil B, Ergebnis | scharfes Beispiel, vier Regeln × drei Reihenfolgen, exakt |
| Experiment 4 | 20 zufällige Instanzen mit je 8 unabhängigen Verteilungen (Exponential, Zwei-Punkt, Gleich), je 50 000 Stichproben; Schwelle E[max]/2 in drei Reihenfolgen |
| Experiment 5 | n = 2 bis 100 unabhängige, gleichverteilte Angebote in (0, 1]; optimal gegen die beiden Schwellen, exakt |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen in `tests/test_claims.py` nachgerechnet. Zahlen aus gezogenen Folgen sind **Stichproben mit festem Seed** und schwanken um etwa ±0,5 Punkte um den exakten Wert (Experiment 1: 37,6 % bei exakt 37,1 %); exakt gerechnete Zahlen sind als „exakt“ gekennzeichnet.

| Frage | Befund |
|---|---|
| Klassisch (n = 100, k = 37, zufällig)? | Exakt **37,1 %**, gezogen 37,1 % (4000 Folgen); das gewählte Angebot ist im Mittel das 0,81-Fache des besten und liegt in 63 % der Fälle unter den besten 5. |
| Bestes k bei wachsendem n (exakt)? | n = 3 / 10 / 100 / 1000 / 10 000: k = 1 / 3 / 37 / 368 / 3679, Erfolg 50,0 % / 39,9 % / 37,1 % / 36,8 % / 36,8 %. Der Erfolg fällt von oben gegen 1/e = 36,79 %, der Anteil k/n liegt bei n = 10 000 bei 0,3679. |
| Aufsteigend sortierte Folge? | Der Beste wird in **0,0 %** der Fälle gewählt (Preset „Aufsteigend sortiert“): Jedes Angebot ist besser als alle vorherigen, und die Regel nimmt das erste nach der Wartezeit, hier das 38. Angebot. Das Mittel der Quotienten gewählt/beste ist 0,379. |
| Nur der Anfang sortiert (Experiment 1)? | Erste 25 %: **36,8 %** (die Sortierung liegt in der Wartezeit und schadet nicht); erste 50 %: **0,0 %**, ebenso bei 75 %, 100 % und bei absteigender Folge. Wert/Maximum 0,752 bei 50 %, 0,505 bei 75 %, 0,380 aufsteigend, **0,010** absteigend. Zufällig: 37,6 %. |
| Früh zuschlagen (k = 10, Preset)? | Exakt 23,5 %, gezogen 23,1 %; das gewählte Angebot ist im Mittel das **0,92-Fache** des besten: weniger oft der Beste, aber im Mittel ein besseres Angebot. |
| Kleine Folge (n = 10, k = 3, Preset)? | Exakt **39,9 %**: bei wenigen Angeboten lohnt sich das Warten mehr als der Grenzwert 36,8 % sagt. |
| Was ist das Ziel (Experiment 2)? | k = 0 / 5 / 10 / 20 / 37 / 50 / 70: Erfolg 1,0 % / 15,2 % / 23,3 % / 32,6 % / **37,2 %** / 35,0 % / 25,2 %; Wert/Maximum 0,504 / 0,899 / **0,913** / 0,885 / 0,810 / 0,748 / 0,650; unter den besten 5: 4,8 % / 47,0 % / 62,1 % / **70,5 %** / 63,1 % / 51,9 % / 33,0 %. Wer den Besten will, wartet 37; wer einen hohen Wert will, nur 10; wer unter die besten 5 will, 20. |
| Scharfes Beispiel (ε = 0,01, exakt)? | E[max] = 1,99. Schwelle E[max]/2: **0,5025** des Maximums bei „kleine zuerst“, **1,000** bei „große zuerst“, 0,7513 zufällig. Immer das erste: 0,5025 in jeder Reihenfolge. Der Quotient der Schwelle bei „kleine zuerst“ ist 1/(2 − ε), also scharf gegen 1/2. Bei ε = 0,5: 0,667 / 1,000 / 0,833. |
| Optimale Regel im scharfen Beispiel? | Gleich wie die Schwelle (0,5025 / 1,000 / 0,7513): Schon hier ist die Schwelle in allen drei Reihenfolgen optimal; beide Schwellen (E[max]/2 und Median) stimmen im ganzen Beispiel überein. |
| Zufällige Instanzen (Experiment 4)? | Schwelle E[max]/2: zufällige Reihenfolge im Mittel **0,854** des erwarteten Maximums (Minimum 0,756), aufsteigend nach E[X] 0,800 (Minimum 0,680), absteigend 0,866 (0,759). Weit über der Garantie 0,5, und „kleine zuerst“ schadet auch hier. |
| Unabhängig und gleich verteilt (Experiment 5, exakt)? | n = 2 / 10 / 100: optimal 0,9375 / 0,9472 / **0,9910**, Schwelle E[max]/2 0,9167 / 0,7998 / **0,7550**, Schwelle am Median 0,9053 / 0,7882 / 0,7540. Die Worst-Case-Schwelle lässt bei n = 100 **23,6 Punkte** liegen. |

## Ehrliche Grenzen

- Die Demo ist **nur an erfundenen Angeboten** geprüft: gleichverteilte Werte in Teil A, ein Zwei-Punkt-Beispiel und zufällige Verteilungen in Teil B.
- **Stichproben:** Erfolg bei gewählter Reihenfolge, Wert/Maximum, „unter den besten 5“ und Experiment 4 sind Schätzungen mit festem Seed. Die Erfolgswahrscheinlichkeit bei zufälliger Reihenfolge und alles in Teil B (außer Experiment 4) ist exakt.
- **Wert gegen Rang:** Das Sekretärproblem sieht nur die Rangfolge, das Prophet-Problem die Werte; die Zahlen beider Teile sind nicht vergleichbar, nur die Idee „eine Wahl ohne Zurück“ verbindet sie.
- Der Grenzwert 1/e gilt für große n; bei kleinem n liegt der Erfolg höher (n = 10: 39,9 %).
- Die Garantie 1/2 der Schwelle gilt im **schlechtesten Fall**; das scharfe Beispiel zeigt, dass nichts Besseres für alle Reihenfolgen gilt, im Mittel über zufällige Instanzen liegt die Schwelle weit darüber.
- Die **Primärquellen wurden nicht eingesehen**; die Aussagen beruhen auf geschlossenen Formeln und Durchrechnen. Deshalb gibt es keinen Literaturtest; die Formeln sind aber gegen vollständiges Aufzählen aller Reihenfolgen (n bis 7) geprüft.
- Nicht modelliert: mehrere Zuschläge (Matroid-Sekretärproblem), Angebote mit Kosten fürs Ansehen, abhängige Angebote, Reihenfolgen eines Gegners im Sekretärproblem außer der aufsteigenden und der absteigenden.

## Verifikation

- **Umfang:** siehe `pytest --collect-only -q tests`, rund eineinhalb Minuten lokal.
- **Kern von Hand** (`tests/test_algorithm.py`): Erfolgsformel (n = 3, k = 1: drei von sechs), Wahlregel einzeln und vektorisiert, Kennzahlen, scharfes Beispiel, gleichverteilte Rückwärtsrechnung (n = 2: 5/8).
- **Orakel** (`tests/test_oracle_sekretaer.py`): die Formel gegen das **Aufzählen aller Reihenfolgen** für n = 1 bis 7 und jedes k; das scharfe Beispiel gegen eine getrennte Aufzählung der Ausgänge; Stichproben gegen die Formel (innerhalb 4,5 Standardabweichungen); die gleichverteilten Werte gegen Simulationen mit schleifenweiser Regel.
- **Eigenschaften** (`tests/test_eigenschaften_sekretaer.py`): die Erfolgskurve steigt bis zum besten k und fällt danach; die Wahl hängt nur von der Rangfolge ab; **Online-Eigenschaft** (spätere Angebote ändern die Wahl nicht); gewählt wird ein Rekord oder das letzte; aufsteigende Folge verfehlt den Besten außer bei k = n − 1; absteigende nur ohne Wartezeit; die Schwelle garantiert 0,5 in jeder Reihenfolge und ist scharf (1/(2 − ε)); optimal schlägt alle Regeln.
- **Literatur:** kein Referenzwert-Test (siehe oben, **offen**).
- **Verdikte** (`tests/test_verdicts.py`): jede Zahl und jede Verzweigung der Texte. **Claims** (`tests/test_claims.py`): jede Zahl dieser README.
- **Presets, Layout, Oberfläche:** vollständig, Werte in den Reglergrenzen; Permalink übernimmt gültige Werte und begrenzt ungültige; Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben; AppTest-Rauchtest (jeder Button, jeder Regler an beiden Grenzen, jede Auswahl jedes Auswahlfelds); Footer; Laufzeit-Importe.
- **Browser:** im Desktop-Browser durchgeklickt (Standard, „Aufsteigend sortiert“ und „Prophet: große zuerst“, Seitenleiste, Diagramme, Formeln ohne Darstellungsfehler; die übrigen drei Presets nur im AppTest-Rauchtest). Nicht mit echtem Touch-Gerät getestet.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Teil A und Teil B mit Kernfrage, fünf Experimente, Grenzen, Formeln |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `sek_algorithm.py` | Kern: Erfolgsformel, Wahlregel, Kennzahlen, scharfes Beispiel, gleichverteilte Rückwärtsrechnung |
| `sek_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Farben, Presets |
| `sek_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `sek_layout.py` | Fester Kopf (Stylesheet) |
| `sek_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `sek_scenario.py` | Folgen von Angeboten, teilweise sortiert |
| `sek_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Kern, Randfälle, Orakel, Eigenschaften, Verdikte, Claims, Presets, Layout, Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mehrere Zuschläge** (k-Sekretärproblem, Matroid), Kosten fürs Ansehen, abhängige Angebote.
- **Beweise** der Garantien: belegt sind sie hier durch Aufzählen und Rechnen, nicht durch Herleitung.
- **Lernende Regeln** (Schwelle aus Stichproben, Vorhersagen): ein eigenes Stück der Reihe.
- **Echte Angebotsdaten** und PDF-Export.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
