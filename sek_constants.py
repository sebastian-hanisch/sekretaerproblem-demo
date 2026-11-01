"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Teil A: Sekretärproblem
N_OPTIONS = (10, 20, 50, 100, 200, 500, 1000)
DEFAULT_N = 100
KPCT_MIN, KPCT_MAX, DEFAULT_KPCT = 0, 95, 37          # Wartezeit k in Prozent von n; k = min(n − 1, gerundet)
SORT_MIN, SORT_MAX, SORT_STEP, DEFAULT_SORT = 0, 100, 5, 0   # Anteil der Folge am Anfang, der aufsteigend sortiert ist (0 = zufällige Reihenfolge)
TRIALS = 4000                                         # Ziehungen für die Schätzung (Erfolg, Wert, Rang)
CURVE_POINTS = 25                                     # Stützstellen der Erfolgskurve über k
TOP = 5                                               # „unter den besten 5“

# Teil B: Prophet-Ungleichung, scharfes Beispiel X1 = 1 sicher, X2 = 1/eps mit Wahrscheinlichkeit eps, sonst 0
EPS_OPTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_EPS = 0.01
ORDNUNGEN_B = ("kleine zuerst", "große zuerst", "zufällig")
DEFAULT_ORDNUNG_B = "kleine zuerst"
ORDNUNG_B_HILFE = {
    "kleine zuerst": "Erst die sichere 1, dann das riskante Angebot: die Reihenfolge, die der Regel am meisten schadet.",
    "große zuerst": "Erst das riskante Angebot, dann die sichere 1: die günstigste Reihenfolge.",
    "zufällig": "Die beiden Angebote kommen mit gleicher Wahrscheinlichkeit in einer der beiden Reihenfolgen.",
}
REGELN_B = ("Schwelle E[max]/2", "Schwelle Median des Maximums", "immer das erste", "optimal (Rückwärtsrechnung)")
DEFAULT_REGEL_B = "Schwelle E[max]/2"
REGEL_B_HILFE = {
    "Schwelle E[max]/2": "Das erste Angebot, das mindestens die Hälfte des erwarteten Maximums erreicht; sonst das letzte. Garantiert die Hälfte von E[max] in jeder Reihenfolge.",
    "Schwelle Median des Maximums": "Das erste Angebot, das den Median des Maximums erreicht; sonst das letzte.",
    "immer das erste": "Das erste Angebot nehmen, egal wie hoch.",
    "optimal (Rückwärtsrechnung)": "Die beste Regel für die gegebene Reihenfolge: nimm ein Angebot, wenn es mehr wert ist als das, was man danach im Mittel noch bekommt.",
}
SEED_MAX, DEFAULT_SEED = 999999, 7

# feste Experimente (nie ein einzelner Seed als Befund)
ORDER_N, ORDER_K, ORDER_SHARES = 100, 37, (0, 25, 50, 75, 100)
ORDER_TRIALS = 50000
VALUE_N, VALUE_KS, VALUE_TRIALS = 100, (0, 5, 10, 15, 20, 30, 37, 50, 70), 50000
GROWTH_NS = (3, 5, 10, 20, 50, 100, 1000, 10000)
INSTANCES = 20                                        # zufällige Prophet-Instanzen
INSTANCE_VARS = 8
INSTANCE_SAMPLES = 50000
IID_NS = (2, 3, 5, 10, 25, 50, 100)

COLOR_BEST = "#4c78a8"
COLOR_PICK = "#2F6B65"
COLOR_SEEN = "rgba(150,150,150,0.65)"
COLOR_OTHER = "rgba(150,150,150,0.35)"
COLOR_EXACT = "#4c78a8"
COLOR_MC = "#c0392b"
COLOR_GUARANTEE = "#d97706"
COLORS_B = {"Schwelle E[max]/2": "#2F6B65", "Schwelle Median des Maximums": "#7c3aed", "immer das erste": "#c0392b", "optimal (Rückwärtsrechnung)": "#4c78a8"}

PRESETS = {
    "Klassisch (n = 100, 37 %)": {"n": DEFAULT_N, "kpct": 37, "sort": 0, "eps": DEFAULT_EPS, "ordnung_b": "kleine zuerst", "regel_b": "Schwelle E[max]/2", "seed": DEFAULT_SEED},
    "Aufsteigend sortiert": {"n": DEFAULT_N, "kpct": 37, "sort": 100, "eps": DEFAULT_EPS, "ordnung_b": "kleine zuerst", "regel_b": "Schwelle E[max]/2", "seed": DEFAULT_SEED},
    "Erste Hälfte sortiert": {"n": DEFAULT_N, "kpct": 37, "sort": 50, "eps": DEFAULT_EPS, "ordnung_b": "kleine zuerst", "regel_b": "Schwelle E[max]/2", "seed": DEFAULT_SEED},
    "Früh zuschlagen (k = 10 %)": {"n": DEFAULT_N, "kpct": 10, "sort": 0, "eps": DEFAULT_EPS, "ordnung_b": "kleine zuerst", "regel_b": "Schwelle E[max]/2", "seed": DEFAULT_SEED},
    "Kleine Folge (n = 10)": {"n": 10, "kpct": 30, "sort": 0, "eps": DEFAULT_EPS, "ordnung_b": "kleine zuerst", "regel_b": "Schwelle E[max]/2", "seed": DEFAULT_SEED},
    "Prophet: große zuerst": {"n": DEFAULT_N, "kpct": 37, "sort": 0, "eps": DEFAULT_EPS, "ordnung_b": "große zuerst", "regel_b": "Schwelle E[max]/2", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Klassisch (n = 100, 37 %)": "Hundert Angebote in zufälliger Reihenfolge, die ersten 37 nur ansehen: der Beste wird in etwas mehr als einem Drittel der Fälle gewählt.",
    "Aufsteigend sortiert": "Die Angebote kommen aufsteigend: jedes ist besser als alle vorherigen, die Regel nimmt das 38. und verfehlt den Besten immer.",
    "Erste Hälfte sortiert": "Nur die erste Hälfte der Folge ist aufsteigend sortiert: schon das zerstört die Regel vollständig.",
    "Früh zuschlagen (k = 10 %)": "Nur zehn Angebote ansehen: der Beste wird seltener gefunden, aber das gewählte Angebot ist im Mittel besser.",
    "Kleine Folge (n = 10)": "Zehn Angebote, drei ansehen: der Beste wird in knapp vier von zehn Fällen gewählt (die Formel ist exakt).",
    "Prophet: große zuerst": "Das riskante Angebot kommt zuerst: die Schwellenregel erreicht das volle erwartete Maximum.",
}
