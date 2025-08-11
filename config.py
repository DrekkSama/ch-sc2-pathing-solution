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
LIVE_RISK_WEIGHT = 0.7          # Weight for current session risks
LEARNED_RISK_WEIGHT = 0.3       # Weight for historical risks
TEMPORAL_DECAY_RATE = 0.02      # Risk decay per step (2%)
SUCCESS_REWARD = -5             # Risk reduction for successful passage
EXPLORATION_BONUS = -0.1        # Bonus for under-explored areas
DEATH_PENALTY_LIVE = 50         # Risk increase for current session deaths
DEATH_PENALTY_LEARNED = 20      # Risk increase for historical deaths
MIN_VISITS_FOR_EXPLORATION = 3  # Visits needed before exploration bonus stops