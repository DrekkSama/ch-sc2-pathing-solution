# Purpose: Centralized configuration for maps and dual-layer risk learning
# Key Decisions: Minimal tunables, deterministic defaults, single Data dir
# Limitations: No per-map overrides; tune constants manually
# Configuration for map settings

# Default map name. Change this value to switch the default map easily.
DEFAULT_MAP = "LightShade_Pathing_Gym"

# Optional: list of available maps
AVAILABLE_MAPS = [
    "LightShade_Pathing_0",
    "LightShade_Pathing_1",
    "LightShade_Pathing_1_2",
    "LightShade_Pathing_Gym",
]

# Risk Management Configuration
DATA_DIR = "Data"
LEARNED_RISKS_FILE = "learned_risks.npy"
EXPLORATION_MAP_FILE = "exploration_map.npy"

# Risk System Parameters
LIVE_RISK_WEIGHT = 0.5          # Weight for current session risks
LEARNED_RISK_WEIGHT = 0.5       # Weight for historical risks
TEMPORAL_DECAY_RATE = 0.01      # Risk decay per step (2%)
SUCCESS_REWARD = -5             # Risk reduction for successful passage
EXPLORATION_BONUS = -0.1        # Bonus for under-explored areas
DEATH_PENALTY_LIVE = 50         # Risk increase for current session deaths
DEATH_PENALTY_LEARNED = 20      # Risk increase for historical deaths
MIN_VISITS_FOR_EXPLORATION = 3  # Visits needed before exploration bonus stops

# Damage and enemy presence risk parameters
DAMAGE_RISK_PER_HP = 0.6
DAMAGE_SPREAD_RADIUS = 10
ENEMY_PRESENCE_PENALTY = 0.6
ENEMY_PRESENCE_RADIUS = 10

# Risk update clamps
LIVE_RISK_MAX_ADD_PER_CALL = 2.0  # Cap per-tile increment from a single radial application