"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

GRIDS = {"2x3": (2, 3), "3x3": (3, 3), "2x5": (2, 5), "3x4": (3, 4)}
DEFAULT_GRID = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072, rund 0,5 s); eigenes Netz ebenso
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

# Kontrollvariaten (alle mit exakt bekanntem Erwartungswert): Schlüssel -> Namen der Größen; "alle" nimmt die drei zusammen
CONTROL_NAMES = {"K": "Zahl der haltenden Kanten", "iso": "Zahl der isolierten Knoten", "tree": "ein fester Spannbaum hält ganz"}
CONTROLS = {"K": ("K",), "iso": ("iso",), "tree": ("tree",), "alle": ("K", "iso", "tree")}
CONTROL_LABELS = {"K": "Kantenzahl K", "iso": "isolierte Knoten", "tree": "Spannbaum hält", "alle": "alle drei zusammen"}
DEFAULT_CONTROL = "iso"

P_MIN, P_MAX, DEFAULT_P = 0.50, 0.99, 0.90
N_CHOICES = (30, 100, 300, 1000, 3000)    # Ziehungen je Schätzung
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 1000
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS = 200                                # Wiederholungen je Schätzung (feste Ströme), Rauschen der Varianzverhältnisse 2/sqrt(REPS − 1) ≈ 14 %

# Experimente
SWEEP_N = (30, 100, 300, 1000)
SWEEP_REPS = 300
CHOICE_PS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
BETA_N = (30, 100, 300, 1000)
BETA_REPS = 600
RARE_PS = (0.95, 0.99)
RARE_N = (100, 1000, 10000)
RARE_REPS = 150
CUPED_RHOS = (0.2, 0.5, 0.8)
CUPED_EXPERIMENTS = 20_000
CUPED_ARM = 50

Z95 = 1.959963984540054

COLOR_PLAIN = "#4c78a8"
COLOR_CV = "#2F6B65"
COLOR_ORACLE = "#E08E0B"
COLOR_TRUTH = "#222222"
COLOR_BAD = "#C0392B"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (Voreinstellung)": {"gitter": DEFAULT_GRID, "p": DEFAULT_P, "n": DEFAULT_N, "kontrolle": DEFAULT_CONTROL, "seed": DEFAULT_SEED},
    "Seltener Ausfall": {"gitter": DEFAULT_GRID, "p": 0.99, "n": 3000, "kontrolle": "iso", "seed": DEFAULT_SEED},
    "Die naheliegende Wahl": {"gitter": DEFAULT_GRID, "p": 0.99, "n": 3000, "kontrolle": "K", "seed": DEFAULT_SEED},
    "Kleine Stichprobe, drei Kontrollvariaten": {"gitter": DEFAULT_GRID, "p": 0.70, "n": 30, "kontrolle": "alle", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4, jede Kante hält mit 90 %, 1000 Ziehungen, Kontrollvariate „isolierte Knoten“: ein deutlicher Gewinn.",
    "Seltener Ausfall": "Dasselbe Gitter bei p = 0,99: der Ausfall ist selten, die Zahl isolierter Knoten trifft seinen Mechanismus.",
    "Die naheliegende Wahl": "Derselbe seltene Ausfall mit der Kantenzahl als Kontrollvariate: hier hilft sie praktisch nicht.",
    "Kleine Stichprobe, drei Kontrollvariaten": "Nur 30 Ziehungen und drei Kontrollvariaten: das Schätzen von drei Koeffizienten frisst den Gewinn.",
}
