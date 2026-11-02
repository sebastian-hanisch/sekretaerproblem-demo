"""Sekretärproblem und Prophet – Angebote einholen, ohne zurück zu können - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Reihe „Online-Algorithmen“: Angebote kommen nacheinander, jedes wird sofort genommen oder verworfen.
Teil A kennt nur die Rangfolge der bisherigen Angebote (Sekretärproblem), Teil B kennt die Verteilungen (Prophet-Ungleichung).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sek_algorithm as A
import sek_constants as C
from sek_evaluation import (
    Settings,
    analyse_a,
    analyse_b,
    delta_text,
    growth_rows,
    iid_rows,
    instance_rows,
    order_rows,
    value_rows,
    verdict_a,
    verdict_b,
    verdict_growth,
    verdict_iid,
    verdict_instances,
    verdict_order,
    verdict_value,
)
from sek_layout import dunkles_design, kopf_css
from sek_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from sek_visualization import (
    build_example,
    build_growth,
    build_iid,
    build_instances,
    build_order,
    build_prophet,
    build_success_curve,
    build_value,
)

st.set_page_config(page_title="Sekretärproblem und Prophet – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Ziehe die Folgen …")
def _analysis_a(settings):
    return analyse_a(settings)


@st.cache_data(show_spinner=False)
def _analysis_b(settings):
    return analyse_b(settings)


@st.cache_data(show_spinner="Ziehe 50 000 Folgen je Reihenfolge …")
def _order():
    return order_rows()


@st.cache_data(show_spinner=False)
def _value():
    return value_rows()


@st.cache_data(show_spinner=False)
def _growth():
    return growth_rows()


@st.cache_data(show_spinner="Rechne 20 zufällige Instanzen …")
def _instances():
    return instance_rows()


@st.cache_data(show_spinner=False)
def _iid():
    return iid_rows()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sek_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎩 Sekretärproblem und Prophet – Angebote einholen, ohne zurück zu können")
st.markdown(
    """
**Fünftes Stück der Reihe „Online-Algorithmen“** (Entscheiden ohne Kenntnis der Zukunft). Angebote kommen nacheinander; jedes wird **sofort genommen oder für immer verworfen**. Es gibt zwei Fassungen, was man über die Angebote weiß:

- **Teil A, Sekretärproblem:** Man kennt nur die **Rangfolge** der bisherigen Angebote (ist dieses das beste bisher?), sonst nichts. Die Reihenfolge ist zufällig. Die Regel „die ersten k nur ansehen, dann das erste nehmen, das besser ist als alle bisherigen“ findet den Besten mit Wahrscheinlichkeit gegen **1/e ≈ 0,37**.
- **Teil B, Prophet-Ungleichung:** Man kennt die **Verteilungen** unabhängiger Angebote, aber nicht ihre Werte, und die Reihenfolge bestimmt ein Gegner. Eine einzige **Schwelle** (die Hälfte des erwarteten Maximums) sichert **mindestens die Hälfte** des Wertes, den ein Prophet erreichen würde.

Beide Teile haben ihre eigenen Regler in der Seitenleiste. Weiter unten: die **Kernfrage** je Teil, Experimente (**Reihenfolge**, **Ziel**, **wachsende Zahl**, **zufällige Instanzen**, **unabhängig gleich verteilt**), **🚧 Wo die Annahmen enden**, **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktionieren Sekretärproblem und Prophet", expanded=True):
    st.markdown(
        """
1. **Teil A:** n Angebote kommen in zufälliger Reihenfolge, größer ist besser. Nach jedem Angebot weiß man nur, ob es das beste bisher ist. Die Regel schaut sich die ersten **k** nur an und nimmt dann das erste, das besser ist als alle bisher gesehenen (kommt keines, das letzte).
2. **Warum 37 %:** Wer zu früh zuschlägt, nimmt oft ein mittelmäßiges Angebot; wer zu lange wartet, lässt den Besten verstreichen. Das Optimum liegt bei k ≈ n/e, der Erfolg bei ≈ 1/e.
3. **Die Annahme „zufällige Reihenfolge“ trägt alles:** Kommen die Angebote aufsteigend, ist jedes das beste bisher, und die Regel nimmt das erste nach der Wartezeit, das nicht das beste ist (außer bei einer Wartezeit von n − 1). Reicht der sortierte Anfang über die Wartezeit hinaus, ist die Regel sofort wirkungslos.
4. **Teil B:** Die Angebote sind unabhängige Zufallsgrößen mit bekannter Verteilung. Ein Prophet sieht alle Werte und nimmt das größte (Mittel E[max]). Die Schwelle E[max]/2 nimmt das erste Angebot darüber und garantiert **in jeder Reihenfolge** mindestens E[max]/2.
5. **Das scharfe Beispiel:** Ein sicheres Angebot 1 und ein riskantes mit Wert 1/ε und Wahrscheinlichkeit ε. Kommt die sichere 1 zuerst, erreicht die Schwelle nur die Hälfte; kommt das riskante zuerst, das Ganze.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Teil A: Sekretärproblem**")
    n = st.select_slider("Zahl der Angebote n", options=C.N_OPTIONS, key="n_select")
    kpct = st.slider("Nur ansehen: k in % von n", *bounds("kpct_slider"), key="kpct_slider", help="Wie viele Angebote ohne Zuschlag vorbeiziehen. 37 % ist der Optimalwert für große n.")
    sort = st.slider("Davon aufsteigend sortiert, % der Folge", *bounds("sort_slider"), step=C.SORT_STEP, key="sort_slider", help="Die ersten so viel Prozent der Folge kommen aufsteigend; 0 heißt zufällige Reihenfolge, die Annahme der Garantie.")
    st.markdown("**Teil B: Prophet**")
    eps = st.select_slider("Wahrscheinlichkeit ε des riskanten Angebots", options=C.EPS_OPTIONS, key="eps_select", help="Das riskante Angebot hat den Wert 1/ε mit Wahrscheinlichkeit ε, sonst 0; das sichere ist 1.")
    ordnung_b = st.selectbox("Reihenfolge", C.ORDNUNGEN_B, key="ordnung_b_select", help="Welches der beiden Angebote zuerst kommt.")
    st.caption(C.ORDNUNG_B_HILFE[ordnung_b])
    regel_b = st.selectbox("Regel", C.REGELN_B, key="regel_b_select", help="Die Regeln laufen im Vergleich alle mit; die gewählte steht im Ergebnis und im Verdikt.")
    st.caption(C.REGEL_B_HILFE[regel_b])
    st.markdown("**Zufall (Teil A)**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Folgen ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "kpct_slider": int(kpct), "sort_slider": int(sort), "eps_select": float(eps), "ordnung_b_select": ordnung_b, "regel_b_select": regel_b, "seed_input": int(seed)})

settings = Settings(int(n), int(kpct), int(sort), float(eps), ordnung_b, regel_b, int(seed))
a = _analysis_a(settings)
b = _analysis_b(settings)

# --- Teil A: Ergebnis ----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎩 Teil A: Sekretärproblem")
st.caption(f"{n} Angebote · die ersten {a.k} ({kpct} %) nur ansehen · " + ("zufällige Reihenfolge" if sort == 0 else f"die ersten {sort} % aufsteigend sortiert") + f" · {C.TRIALS} gezogene Folgen")
st_a = a.stats
m1, m2, m3, m4 = st.columns(4)
d1, c1 = delta_text(st_a["Erfolg"], a.exact, "die exakte Zahl")
m1.metric("Den Besten gewählt", f"{st_a['Erfolg'] * 100:.1f} %", delta=d1, delta_color=c1, help=f"Anteil der {C.TRIALS} gezogenen Folgen, in denen das gewählte Angebot das beste ist. Verglichen mit der exakten Formel für zufällige Reihenfolge.")
m2.metric("Exakt (zufällige Reihenfolge)", f"{a.exact * 100:.1f} %", delta="Formel", delta_color="off", help="(k/n) · Σ 1/i für i von k bis n − 1; gilt nur bei zufälliger Reihenfolge.")
m3.metric("Gewählt / bestes", f"{st_a['Wert/Maximum']:.3f}", delta="Mittel der Quotienten", delta_color="off", help="Wert des gewählten Angebots geteilt durch den des besten, im Mittel.")
m4.metric(f"Unter den besten {C.TOP}", f"{st_a['unter den besten 5'] * 100:.0f} %", delta="der Folgen", delta_color="off", help=f"Anteil der Folgen, in denen das gewählte Angebot zu den besten {C.TOP} gehört.")
ex = a.example
st.plotly_chart(build_example(ex["values"], a.k, ex["chosen"], ex["best"]), width="stretch", key="example_chart")
st.caption("Eine gezogene Folge: grau die nur angesehenen Angebote, blass die übrigen, grün das gewählte, blau das beste (fällt beides zusammen, ist es grün). Neue Folge mit dem Würfel in der Seitenleiste.")
st.plotly_chart(build_success_curve(*a.curve, a.k, int(n), int(sort)), width="stretch", key="curve_chart")
st.caption("Erfolgswahrscheinlichkeit über der Wartezeit k: exakt für zufällige Reihenfolge, geschätzt für die gewählte. Die gestrichelte Linie ist 1/e.")

st.markdown("### ❓ Kernfrage A: Findet die Regel den Besten?")
kind, text = verdict_a(a)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 1: Was kostet eine sortierte Folge?")
rows_or = _order()
st.caption(f"n = {C.ORDER_N}, k = {C.ORDER_K}, je {C.ORDER_TRIALS} gezogene Folgen, fester Seed. Die ersten x % der Folge aufsteigend sortiert; absteigend ist die ganze Folge umgekehrt sortiert.")
st.plotly_chart(build_order(rows_or), width="stretch", key="order_chart")
kind, text = verdict_order(rows_or)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 2: Der Beste oder ein guter?")
rows_va = _value()
st.caption(f"n = {C.VALUE_N}, zufällige Reihenfolge, je {C.VALUE_TRIALS} gezogene Folgen. Dieselbe Regel, drei Ziele: den Besten finden, einen hohen Wert erzielen, unter den besten {C.TOP} landen.")
st.plotly_chart(build_value(rows_va), width="stretch", key="value_chart")
kind, text = verdict_value(rows_va)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 3: Wächst die Zahl der Angebote?")
rows_gr = _growth()
st.caption("Exakt gerechnet für zufällige Reihenfolge: das beste k und sein Erfolg für n von 3 bis 10 000.")
st.plotly_chart(build_growth(rows_gr), width="stretch", key="growth_chart")
kind, text = verdict_growth(rows_gr)
getattr(st, kind)(text)

st.markdown("---")

# --- Teil B: Ergebnis ----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔮 Teil B: Prophet-Ungleichung")
st.caption(f"Sicheres Angebot 1, riskantes Angebot {1 / eps:g} mit Wahrscheinlichkeit {eps:g} (sonst 0) · Reihenfolge „{ordnung_b}“ · Regel: {regel_b}")
c1m, c2m, c3m = st.columns(3)
c1m.metric("Wert der Regel", f"{b.value:.3f}", delta=f"Reihenfolge „{ordnung_b}“", delta_color="off", help="Erwarteter Wert der gewählten Regel in der gewählten Reihenfolge; exakt gerechnet.")
c2m.metric("Erwartetes Maximum (Prophet)", f"{b.emax:.3f}", delta="Referenz", delta_color="off", help="Mittel des größten Wertes; der Prophet sieht alle Werte vorher.")
c3m.metric("Anteil am Prophet", f"{b.ratio:.3f}", delta="Garantie der Schwelle: 0,5", delta_color="off", help="Wert der Regel geteilt durch E[max].")
st.plotly_chart(build_prophet(b.table), width="stretch", key="prophet_chart")
st.caption("Alle Regeln in allen Reihenfolgen, als Anteil des erwarteten Maximums. Die Schwellenregel liegt nie unter der gestrichelten Garantie 0,5; „kleine zuerst“ ist ihre ungünstigste Reihenfolge. In diesem Beispiel stimmen beide Schwellen und die optimale Regel überein; sie unterscheiden sich erst in Experiment 5.")

st.markdown("### ❓ Kernfrage B: Wie viel vom Prophetenwert erreicht die Regel?")
kind, text = verdict_b(b)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 4: Zufällige Instanzen")
rows_in = _instances()
st.caption(f"{C.INSTANCES} zufällige Instanzen mit {C.INSTANCE_VARS} unabhängigen Verteilungen (Exponential, Zwei-Punkt, Gleich), je {C.INSTANCE_SAMPLES} Stichproben, fester Seed. Schwelle E[max]/2; Zahl = Wert geteilt durch E[max].")
st.plotly_chart(build_instances(rows_in), width="stretch", key="instances_chart")
kind, text = verdict_instances(rows_in)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 5: Unabhängig und gleich verteilt")
rows_ii = _iid()
st.caption("n unabhängige, gleichverteilte Angebote in (0, 1], exakt gerechnet. Die optimale Rückwärtsrechnung gegen die beiden Schwellen.")
st.plotly_chart(build_iid(rows_ii), width="stretch", key="iid_chart")
kind, text = verdict_iid(rows_ii)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zufällige Reihenfolge** (Teil A) | Bei aufsteigender Folge verfehlt die Regel den Besten (außer bei einer Wartezeit von n − 1); reicht der sortierte Anfang über die Wartezeit hinaus, kostet das alles (Experiment 1). | Online-Algorithmen mit Gegner, Paging |
| **Nur ein Angebot wählen** | Mehrere Angebote oder Zuschläge (Matroid-Sekretärproblem) brauchen andere Regeln; nicht modelliert. | Online-Allokation |
| **Die Verteilungen sind bekannt** (Teil B) | Ohne Verteilung gilt die Schwelle nicht; mit einer einzigen Stichprobe je Verteilung gibt es verwandte Garantien. | Vorhersagen |
| **Unabhängige Angebote** (Teil B) | Bei Abhängigkeiten gilt die Garantie ½ nicht allgemein. | – |
| **Der Wert eines Angebots ist bei Ankunft bekannt** (Teil B) | Im Sekretärproblem kennt man nur die Rangfolge; die Fassungen sind verwandt, aber nicht gleich. | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        f"""
**Annahmen.** Teil A: die Werte sind gleichverteilt in (0, 1) und verschieden; die Regel sieht nur, ob ein Angebot besser ist als alle bisherigen. Die Kennzahlen „Wert/Maximum“ und „unter den besten {C.TOP}“ sind **Schätzungen** über {C.TRIALS} gezogene Folgen (Experimente 1 und 2: {C.ORDER_TRIALS}); die Erfolgswahrscheinlichkeit bei zufälliger Reihenfolge ist exakt aus der Formel. Teil B: zwei Angebote, das sichere hat den Wert 1, das riskante {1 / eps:g} mit Wahrscheinlichkeit {eps:g}; die Zahlen dort sind exakt, nicht geschätzt. Experiment 4 ist eine Stichprobe mit festem Seed.

**Grenzen.** Die Demo ist nur an erfundenen Angeboten geprüft. Der Grenzwert 1/e gilt für große n; bei kleinem n liegt der beste erreichbare Erfolg höher (bei n = 10 und k = 3 bei {A.p_success(10, 3) * 100:.1f} %). Die Garantie ½ der Schwelle gilt im schlechtesten Fall und ist nur im Grenzfall ε → 0 scharf (die App erlaubt ε ab {min(C.EPS_OPTIONS):g}); im Mittel über zufällige Instanzen liegt sie weit darüber (Experiment 4). Die Rückwärtsrechnung in Teil B kennt die Reihenfolge, die Schwellenregel nicht.

**Bekannte Probleme und Unsicherheiten.** Bei Gleichstand der Qualität (hier ausgeschlossen, da die Werte verschieden sind) wäre die Regel nicht eindeutig. Die Schätzungen schwanken mit dem Seed um einige Zehntelprozentpunkte; die Verdikte und README-Zahlen gelten für feste Seeds.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Teil A: $n$ Angebote mit verschiedenen Werten kommen in zufälliger Reihenfolge. Nach dem $t$-ten Angebot ist nur bekannt, ob es das beste unter den ersten $t$ ist. Gesucht ist eine Regel, die das beste Angebot mit möglichst großer Wahrscheinlichkeit wählt. Teil B: Angebote $X_1, \dots, X_n \ge 0$ sind unabhängig mit bekannten Verteilungen und kommen in einer vom Gegner gewählten Reihenfolge.

| Symbol | Bedeutung |
|---|---|
| $n$, $k$ | Zahl der Angebote und der nur angesehenen |
| $P(n, k)$ | Wahrscheinlichkeit, das beste Angebot zu wählen |
| $X_i$ | Angebot $i$ (Teil B), Zufallsgröße |
| $\mathbb{E}[\max_i X_i]$ | Wert des Propheten |
| $T$ | Schwelle der Regel |

**Sekretärregel.** Die ersten $k$ ansehen, dann das erste Angebot wählen, das besser ist als alle bisherigen (sonst das letzte). Für $k \ge 1$ gilt
$P(n, k) = \dfrac{k}{n} \sum_{i=k}^{n-1} \dfrac{1}{i}$.
Für $n \to \infty$ und $k/n \to x$ geht das gegen $-x \ln x$, mit Maximum $1/e$ bei $x = 1/e$.

**Prophet-Ungleichung.** Wählt man das erste $X_i \ge T$ mit $T = \tfrac{1}{2} \mathbb{E}[\max_i X_i]$ (sonst das letzte), so ist der erwartete Wert mindestens $\tfrac{1}{2} \mathbb{E}[\max_i X_i]$, in jeder Reihenfolge. Die Konstante $\tfrac{1}{2}$ ist scharf: $X_1 = 1$ und $X_2 = 1/\varepsilon$ mit Wahrscheinlichkeit $\varepsilon$ (sonst 0), die 1 zuerst, ergibt den Quotienten $\tfrac{1}{2 - \varepsilon}$ (der Prophet erreicht $\mathbb{E}[\max] = 2 - \varepsilon$, die Regel nur $1$), im Grenzfall $\varepsilon \to 0$ also $\tfrac{1}{2}$.

**Unabhängig und gleich verteilt.** Der optimale Wert $v_m$ bei $m$ noch ausstehenden Angeboten gleichverteilt in $(0, 1]$ erfüllt $v_1 = \tfrac{1}{2}$ und $v_{m+1} = \tfrac{1 + v_m^2}{2}$ (nimm das Angebot, wenn es mehr wert ist als $v_m$).

**Literatur (nicht nachgebaut; Volltexte nicht eingesehen).** Das Sekretärproblem mit der Regel 1/e wird E. B. Dynkin (1963) zugeschrieben; T. S. Ferguson, *Who Solved the Secretary Problem?*, Statistical Science 4 (1989), beschreibt Geschichte und Lösung. Die Prophet-Ungleichung geht auf U. Krengel und L. Sucheston zurück, die Schwellenform auf E. Samuel-Cahn (1984). Hier sind die Aussagen durch vollständiges Durchrechnen kleiner Fälle, geschlossene Formeln und Stichproben belegt, nicht bewiesen.
        """
    )

st.markdown("---")
st.caption(
    "Verwandte Demos: [Ski-Rental](https://github.com/sebastian-hanisch/ski-rental-demo), [Paging](https://github.com/sebastian-hanisch/paging-demo), "
    "[Online-Packen](https://github.com/sebastian-hanisch/online-packen-demo) und [Online-Allokation](https://github.com/sebastian-hanisch/online-allokation-demo) (weitere Stücke der Reihe), "
    "Buchungsannahme mit Gewichtslimit (`uld-buchungsannahme-demo`, eine Ressource und bekannte Verteilung), Routenresilienz (`routenresilienz-demo`, Entscheiden unter Unsicherheit ohne Wiederholung), "
    "[Online-Matching](https://sebastianhanisch-online-matching-demo.streamlit.app/) (Zuordnen ohne Kenntnis der Zukunft)."
)
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Online-Algorithmen: Entscheiden ohne Zukunft](https://sebastianhanisch.net/konzepte-online-algorithmen.html)."
)
