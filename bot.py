from sc2.bot_ai import BotAI, Race
from sc2.data import Result

from sc2.units import Units
from sc2.unit import Unit

import json
import datetime
import os
import traceback
from typing import Dict, List, Optional, Tuple, Any
from sc2.ids.unit_typeid import UnitTypeId

import numpy as np
import random
import math

from sc2.position import Point2, Point3
from map_analyzer import MapData
import config

class RiskManager:
    """Manages dual-layer risk system with live and learned risk maps"""
    
    def __init__(self, map_width: int, map_height: int, map_name: str = "default", data_dir: str = None):
        self.map_width = map_width
        self.map_height = map_height
        self.map_name = map_name
        self.data_dir = data_dir or config.DATA_DIR
        
        # Initialize risk maps
        self.live_risk_map = np.zeros((map_width, map_height), dtype=np.float32)
        self.learned_risk_map = np.zeros((map_width, map_height), dtype=np.float32)
        self.exploration_map = np.zeros((map_width, map_height), dtype=np.int32)
        
        # Ensure data directory exists
        self.ensure_data_dir()
        
        # Load persistent data
        self.load_persistent_data()
        
        print(f"[RISK] RiskManager initialized with {map_width}x{map_height} maps for {map_name}")
    
    def ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                print(f"[RISK] Created data directory: {self.data_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create data directory: {e}")
    
    def load_persistent_data(self):
        """Load learned risk map and exploration data from previous sessions"""
        try:
            # Use map-specific file names to avoid coordinate system mismatches
            learned_filename = f"learned_risks_{self.map_name}_{self.map_width}x{self.map_height}.npy"
            exploration_filename = f"exploration_map_{self.map_name}_{self.map_width}x{self.map_height}.npy"
            
            learned_path = os.path.join(self.data_dir, learned_filename)
            if os.path.exists(learned_path):
                loaded_data = np.load(learned_path)
                if loaded_data.shape == (self.map_width, self.map_height):
                    self.learned_risk_map = loaded_data
                    print(f"[RISK] Loaded learned risk map from {learned_path} - shape {loaded_data.shape}")
                    print(f"[RISK] Max learned risk: {loaded_data.max():.2f}, Non-zero entries: {np.count_nonzero(loaded_data)}")
                else:
                    print(f"[RISK] Map dimension mismatch: file {loaded_data.shape} vs current {(self.map_width, self.map_height)}")
                    print(f"[RISK] Starting fresh learned risk map")
            else:
                print(f"[RISK] No learned risk file found for {self.map_name}, starting fresh")
            
            exploration_path = os.path.join(self.data_dir, exploration_filename)
            if os.path.exists(exploration_path):
                loaded_data = np.load(exploration_path)
                if loaded_data.shape == (self.map_width, self.map_height):
                    self.exploration_map = loaded_data
                    print(f"[RISK] Loaded exploration map from {exploration_path}")
                else:
                    print(f"[RISK] Exploration map dimension mismatch, starting fresh")
            else:
                print(f"[RISK] No exploration file found for {self.map_name}, starting fresh")
                
        except Exception as e:
            print(f"[ERROR] Failed to load persistent data: {e}")
    
    def save_persistent_data(self):
        """Save learned risk map and exploration data for future sessions"""
        try:
            # Use map-specific file names to avoid coordinate system mismatches
            learned_filename = f"learned_risks_{self.map_name}_{self.map_width}x{self.map_height}.npy"
            exploration_filename = f"exploration_map_{self.map_name}_{self.map_width}x{self.map_height}.npy"
            
            learned_path = os.path.join(self.data_dir, learned_filename)
            np.save(learned_path, self.learned_risk_map)
            
            exploration_path = os.path.join(self.data_dir, exploration_filename)
            np.save(exploration_path, self.exploration_map)
            
            print(f"[RISK] Saved persistent data to {self.data_dir} for {self.map_name}")
            print(f"[RISK] Learned risk stats: max={self.learned_risk_map.max():.2f}, non-zero={np.count_nonzero(self.learned_risk_map)}")
        except Exception as e:
            print(f"[ERROR] Failed to save persistent data: {e}")
    
    def get_combined_risk(self, x: int, y: int) -> float:
        """Get combined risk value for a position"""
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return 100.0  # High risk for out-of-bounds
        
        live_risk = self.live_risk_map[x, y] * config.LIVE_RISK_WEIGHT
        learned_risk = self.learned_risk_map[x, y] * config.LEARNED_RISK_WEIGHT
        
        # Exploration bonus for under-explored areas
        exploration_bonus = 0
        if self.exploration_map[x, y] < config.MIN_VISITS_FOR_EXPLORATION:
            exploration_bonus = config.EXPLORATION_BONUS
        
        return max(0, live_risk + learned_risk + exploration_bonus)
    
    def add_death_risk(self, position: Point2, is_current_session: bool = True):
        """Add risk at death location"""
        x, y = int(position.x), int(position.y)
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return
        
        if is_current_session:
            # High immediate risk for current session deaths
            self.live_risk_map[x, y] = min(100, self.live_risk_map[x, y] + config.DEATH_PENALTY_LIVE)
            # Also add to learned map for persistence
            self.learned_risk_map[x, y] = min(100, self.learned_risk_map[x, y] + config.DEATH_PENALTY_LEARNED)
        else:
            # Historical death from loaded data
            self.learned_risk_map[x, y] = min(100, self.learned_risk_map[x, y] + config.DEATH_PENALTY_LEARNED)
        
        # Spread risk in a small radius around death location
        radius = 3
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= radius:
                        # Falloff based on distance
                        falloff = max(0, 1 - distance / radius)
                        penalty = (config.DEATH_PENALTY_LIVE if is_current_session else config.DEATH_PENALTY_LEARNED) * falloff * 0.3
                        
                        if is_current_session:
                            self.live_risk_map[nx, ny] = min(100, self.live_risk_map[nx, ny] + penalty)
                        self.learned_risk_map[nx, ny] = min(100, self.learned_risk_map[nx, ny] + penalty * 0.5)
    
    def _apply_radial(self, x: int, y: int, base_penalty: float, radius: int, live: bool = True, learned_scale: float = 0.0) -> None:
        """Apply a radial penalty with linear falloff. Optionally add a fraction to learned map."""
        if base_penalty <= 0 or radius <= 0:
            return
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    d = math.hypot(dx, dy)
                    if d <= radius:
                        falloff = max(0.0, 1.0 - (d / radius))
                        add = max(0.0, base_penalty * falloff)
                        # Clamp per-call contribution to avoid spikes when many sources overlap
                        add = min(add, float(config.LIVE_RISK_MAX_ADD_PER_CALL))
                        if live:
                            self.live_risk_map[nx, ny] = min(100.0, self.live_risk_map[nx, ny] + add)
                        if learned_scale > 0.0:
                            self.learned_risk_map[nx, ny] = min(100.0, self.learned_risk_map[nx, ny] + add * learned_scale)

    def add_damage_risk(self, position: Point2, damage_amount: float) -> None:
        """Increase live risk around where damage was taken. Does not persist to learned map."""
        x, y = int(position.x), int(position.y)
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return
        magnitude = max(0.0, float(damage_amount)) * float(config.DAMAGE_RISK_PER_HP)
        self._apply_radial(x, y, magnitude, int(config.DAMAGE_SPREAD_RADIUS), live=True, learned_scale=0.0)

    def add_enemy_presence_risk(self, position: Point2) -> None:
        """Add small live risk around visible enemy positions each frame."""
        x, y = int(position.x), int(position.y)
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return
        self._apply_radial(x, y, float(config.ENEMY_PRESENCE_PENALTY), int(config.ENEMY_PRESENCE_RADIUS), live=True, learned_scale=0.0)

    def add_success_reward(self, position: Point2):
        """Reduce risk when probe successfully passes through an area"""
        x, y = int(position.x), int(position.y)
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return
        
        # Reduce risk and increment exploration count
        self.live_risk_map[x, y] = max(0, self.live_risk_map[x, y] + config.SUCCESS_REWARD)
        self.exploration_map[x, y] += 1
    
    def apply_temporal_decay(self):
        """Apply time-based decay to live risk map"""
        self.live_risk_map *= (1 - config.TEMPORAL_DECAY_RATE)
        # Keep values above a small threshold to prevent floating point issues
        self.live_risk_map[self.live_risk_map < 0.01] = 0
    
    def get_risk_map_for_pathfinding(self) -> np.ndarray:
        """Get combined risk map for use in pathfinding"""
        combined_map = np.zeros((self.map_width, self.map_height), dtype=np.float32)
        
        for x in range(self.map_width):
            for y in range(self.map_height):
                combined_map[x, y] = self.get_combined_risk(x, y)
        
        return combined_map

class FailureLogger:
    """Handles structured logging of failures for adaptive learning"""
    
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.DATA_DIR
        self.failure_log = []
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = f"failure_log_{self.session_timestamp}.json"
        self.ensure_log_dir()
        
        # Statistics for analysis
        self.stats = {
            "total_deaths": 0,
            "total_pathfinding_failures": 0,
            "high_risk_areas": [],
            "common_failure_points": {}
        }
    
    def ensure_log_dir(self):
        """Create logs directory if it doesn't exist"""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        except Exception as e:
            print(f"[ERROR] Failed to ensure log directory: {e}")
            
            # Try an alternative approach - use absolute path
            try:
                alternative_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
                if not os.path.exists(alternative_dir):
                    os.makedirs(alternative_dir)
                self.log_dir = alternative_dir
            except Exception as e:
                print(f"[ERROR] Failed alternative log directory: {e}")
    
    def log_probe_death(self, position: Point2, context: Dict[str, Any]):
        """Log probe death with context"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Format position for JSON serialization
        pos_dict = {"x": position.x, "y": position.y}
        
        # Process path history for JSON serialization
        path_history = []
        if "path_history" in context:
            path_history = [{"x": p.x, "y": p.y} for p in context["path_history"]]
            context.pop("path_history")
        
        # Process enemy units for JSON serialization
        enemies = []
        if "enemy_units" in context:
            for unit in context["enemy_units"]:
                enemies.append({
                    "type": str(unit.type_id),
                    "position": {"x": unit.position.x, "y": unit.position.y}
                })
            context.pop("enemy_units")
        
        # Create the log entry
        log_entry = {
            "type": "probe_death",
            "timestamp": timestamp,
            "position": pos_dict,
            "path_history": path_history,
            "enemy_units": enemies,
            "context": context
        }
        
        self.failure_log.append(log_entry)
        self.stats["total_deaths"] += 1
        
        # Track common failure points
        pos_key = f"{int(position.x)},{int(position.y)}"
        if pos_key in self.stats["common_failure_points"]:
            self.stats["common_failure_points"][pos_key] += 1
        else:
            self.stats["common_failure_points"][pos_key] = 1
        
        # Update high risk areas if death count exceeds threshold
        if self.stats["common_failure_points"][pos_key] >= 2:
            if pos_key not in [f"{int(p.x)},{int(p.y)}" for p in self.stats["high_risk_areas"]]:
                self.stats["high_risk_areas"].append(position)
        
        # Save to file
        self.save_to_file()
        
        return log_entry
    
    def log_pathfinding_failure(self, start: Point2, goal: Point2, context: Dict[str, Any]):
        """Log pathfinding failure with context"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Format positions for JSON serialization
        start_dict = {"x": start.x, "y": start.y}
        goal_dict = {"x": goal.x, "y": goal.y}
        
        # Create the log entry
        log_entry = {
            "type": "pathfinding_failure",
            "timestamp": timestamp,
            "start": start_dict,
            "goal": goal_dict,
            "context": context
        }
        
        self.failure_log.append(log_entry)
        self.stats["total_pathfinding_failures"] += 1
        
        # Save to file
        self.save_to_file()
        
        return log_entry
    
    def save_to_file(self):
        """Save the current failure log to a JSON file"""
        try:
            # Ensure directory exists
            self.ensure_log_dir()
            
            # Full path to log file
            file_path = os.path.join(self.log_dir, self.session_file)
            
            # Save log and stats to JSON file
            with open(file_path, 'w') as f:
                # Convert Point2 objects to serializable dictionaries
                stats_copy = self.stats.copy()
                if "high_risk_areas" in stats_copy:
                    stats_copy["high_risk_areas"] = [
                        {"x": point.x, "y": point.y} for point in stats_copy["high_risk_areas"]
                    ]
                
                json.dump({
                    "failure_log": self.failure_log,
                    "stats": stats_copy
                }, f, indent=2)
                
            print(f"Saved failure log to {file_path}")
                
        except Exception as e:
            print(f"Error saving failure log to file: {e}")
            traceback.print_exc()
    
    def get_high_risk_areas(self):
        """Return high risk areas for pathfinding avoidance with adaptive learning"""
        # Get base high risk areas that have been tracked in stats
        high_risk_areas = self.stats["high_risk_areas"].copy()
        
        # Analyze failure log to identify additional risk areas that might not be in the stats yet
        failure_count_threshold = 1  # Lower threshold for adaptive learning
        
        # Build a position frequency map from recent failures (last 10 events)
        position_frequencies = {}
        recent_failures = self.failure_log[-10:] if len(self.failure_log) > 10 else self.failure_log
        
        for entry in recent_failures:
            # For probe deaths, add the death position
            if entry["type"] == "probe_death":
                pos_key = f"{int(entry['position']['x'])},{int(entry['position']['y'])}"
                position_frequencies[pos_key] = position_frequencies.get(pos_key, 0) + 1
            
            # For pathfinding failures, analyze recent path history
            elif entry["type"] == "pathfinding_failure":
                # Add the goal position as problematic
                goal_key = f"{int(entry['goal']['x'])},{int(entry['goal']['y'])}"
                position_frequencies[goal_key] = position_frequencies.get(goal_key, 0) + 1
                
                # Check if there's path history in the context
                if "path_history" in entry.get("context", {}) and entry["context"]["path_history"]:
                    # Add the last few points of the path history (likely where the probe got stuck)
                    path_history = entry["context"]["path_history"]
                    if isinstance(path_history, list) and path_history:
                        for point in path_history[-3:]:  # Last 3 points
                            if isinstance(point, dict) and "x" in point and "y" in point:
                                pos_key = f"{int(point['x'])},{int(point['y'])}"
                                position_frequencies[pos_key] = position_frequencies.get(pos_key, 0) + 0.5  # Lower weight
        
        # Add positions that exceed threshold to high risk areas
        for pos_key, count in position_frequencies.items():
            if count >= failure_count_threshold:
                x, y = map(float, pos_key.split(','))
                new_high_risk = Point2((x, y))
                
                # Check if this point is already in high_risk_areas
                if not any(existing.distance_to(new_high_risk) < 3.0 for existing in high_risk_areas):
                    high_risk_areas.append(new_high_risk)
                    # Also add to persistent stats for future reference
                    if new_high_risk not in self.stats["high_risk_areas"]:
                        self.stats["high_risk_areas"].append(new_high_risk)
        
        return high_risk_areas


GREEN = Point3((0, 255, 0))
RED = Point3((255, 0, 0))
BLUE = Point3((0, 0, 255))


class PathfindingProbe(BotAI):
    NAME: str = "PathfindingProbe"
    RACE: Race = Race.Protoss
    
    def __init__(self):
        super().__init__()
        self.map_data = None
        self.path = []
        self.current_path = []
        self.current_waypoint_index = 0
        self.loop_count = 0
        self.path_update_interval = 10  # Update path every 10 steps
        
        # Unit tracking
        self.probe_tag = None
        self.p0 = None
        self.target = None
        self._last_total_hp = None
        
        # Risk management - dual layer system
        self.death_log = []  # Keep for backward compatibility
        self.enemy_units = []
        self.risk_map = None
        self.risk_manager = None  # Will be initialized in on_start when map size is known
        
        # Failure logging system for adaptive learning
        self.failure_logger = FailureLogger()
        self.path_history = []  # Store recent path history for failure context
        self.max_path_history = 20  # Maximum number of recent points to store
        
        # Initialize waypoints as empty, will be set in on_start with map-relative positions
        self.waypoints = []
        self.waypoint_radius = 0.3  # Fraction of map size for waypoint radius
        
        # Debug/visualization
        self.unvisited_positions = []
        self.scout_destination = None
        self.last_scout_position = None
        self.influence_grid = None
        self.direct_distance = 0
        self.visited_positions = {}
        self.path_length = 0
        self.show_debug = True  # Toggle debug visualization
        
        # Initialize with default values that will be updated in on_start
        self.map_center = Point2((0, 0))
        self.map_radius = 0

    async def on_start(self):
        print("[INIT] Starting bot initialization...")
        
        # Initialize map data
        self.map_data = MapData(self, loglevel="INFO", arcade=True)
        grid: np.ndarray = self.map_data.get_pyastar_grid()
        
        # Initialize dual-layer risk management system
        map_width, map_height = grid.shape
        map_name = self.game_info.map_name.replace(" ", "_")  # Clean map name for filenames
        self.risk_manager = RiskManager(map_width, map_height, map_name)
        
        # Initialize risk map with zeros (same shape as grid) - keep for backward compatibility
        self.risk_map = np.zeros(grid.shape, dtype=float)
        
        # Initialize influence grid (1 = walkable, 10 = unwalkable)
        self.influence_grid = np.ones(grid.shape, dtype=np.int32) * 10
        self.influence_grid[grid == 1] = 1  # Set walkable areas to 1
        
        # Calculate map dimensions for waypoint placement
        self.map_center = self.game_info.map_center
        map_radius = min(
            self.game_info.playable_area.width,
            self.game_info.playable_area.height
        ) * 0.5 * 0.8  # 80% of half the smaller dimension
        
        # Set the waypoints to the specific coordinates provided
        waypoints_unsorted = [
            Point2((84.90, 109.94)),  # Area 1
            Point2((60.45, 132.74)),  # Area 2
            Point2((47.20, 108.33)),  # Area 3
            Point2((107.07, 112.04)), # Area 4
            Point2((137.66, 136.96)), # Area 5
            Point2((134.84, 69.69))   # Area 6
        ]
        
        # Wait for workers to be available
        print("[INIT] Waiting for workers...")
        while not self.workers:
            await self.client.step(8)  # Step forward a bit to let the game start
            
        # Initialize probe
        worker = self.workers.first
        if worker:
            self.probe_tag = worker.tag
            self.p0 = worker.position
            print(f"[INIT] Found worker with tag {self.probe_tag} at {self.p0}")
            # Initialize baseline HP+shield for damage detection
            try:
                self._last_total_hp = float(worker.health + worker.shield)
            except Exception:
                self._last_total_hp = None
        else:
            print("[ERROR] No workers found!")
            return
        
        # Sort waypoints by distance to probe's starting position for optimal navigation
        self.waypoints = sorted(waypoints_unsorted, key=lambda wp: self.p0.distance_to(wp))
        print(f"[INIT] Waypoints sorted by distance from starting position {self.p0}:")
        for i, wp in enumerate(self.waypoints):
            distance = self.p0.distance_to(wp)
            print(f"  {i+1}. {wp} (distance: {distance:.2f})")
        
        # Set initial target to first waypoint
        if self.waypoints:
            self.target = self.waypoints[0]
            print(f"[INIT] First waypoint set to {self.target}")
        else:
            print("[ERROR] No waypoints defined!")
            
        # Print waypoints for debugging
        print("[INIT] Waypoints:")
        for i, wp in enumerate(self.waypoints):
            print(f"  {i+1}. {wp}")
            
        # Initialize path
        self.current_path = []
        self.path = [self.p0]
        
        print("[INIT] Bot initialization complete")
        

    async def on_step(self, iteration: int):
        # Safely get the probe - handle case when it might have died
        probe = None
        if self.probe_tag:
            try:
                probe = self.units.by_tag(self.probe_tag)
            except KeyError:
                # Probe with this tag no longer exists
                print(f"[DEATH] Probe with tag {self.probe_tag} no longer exists")
                self.probe_tag = None  # Clear the tag since it's invalid
        
        # Check if we need to handle probe death or get a new probe
        if self.probe_tag is None and self.workers:
            # Probe died, handle death and get new probe
            print(f"[DEATH] Handling probe death at cycle {iteration}, game time {self.time:.1f}")
            self.handle_probe_death()
            
            # Force save logs immediately
            try:
                print("[LOG] Forcing log save after probe death")
                self.failure_logger.save_to_file()
            except Exception as e:
                print(f"[ERROR] Failed to save logs: {str(e)}")
                traceback.print_exc()
                
            probe = self.workers.first
            if probe:
                self.probe_tag = probe.tag
                print(f"[INFO] Probe respawned at {probe.position}")
                # Reset baseline HP+shield on new probe
                try:
                    self._last_total_hp = float(probe.health + probe.shield)
                except Exception:
                    self._last_total_hp = None
        elif not probe and self.workers:
            # No probe assigned yet, get the first worker
            probe = self.workers.first
            if probe:
                self.probe_tag = probe.tag
                print(f"[INFO] Assigned new probe: {probe.tag} at {probe.position}")
                # Initialize baseline HP+shield
                try:
                    self._last_total_hp = float(probe.health + probe.shield)
                except Exception:
                    self._last_total_hp = None
        
        if not probe:
            print("[WARNING] No probe found to control")
            return
            
        # Update seen enemy set
        self.enemy_units = self.enemy_units | self.all_enemy_units

        # Damage-aware risk update (before recomputing combined risk)
        if self.risk_manager and probe:
            try:
                current_total = float(probe.health + probe.shield)
                if self._last_total_hp is None:
                    self._last_total_hp = current_total
                else:
                    delta = current_total - self._last_total_hp
                    if delta < -0.05:  # Took damage
                        self.risk_manager.add_damage_risk(probe.position, -delta)
                    self._last_total_hp = current_total
            except Exception as e:
                print(f"[RISK] Damage tracking error: {str(e)[:80]}")

        self.update_risk_map()
        
        # Print debug info periodically
        if iteration % 50 == 0:
            print(f"[DEBUG] Probe at {probe.position}, health: {probe.health}")
            print(f"[DEBUG] Current waypoint: {self.waypoints[self.current_waypoint_index]}")
            print(f"[DEBUG] Game state - Time: {self.time:.1f}, Supply: {self.supply_used}/{self.supply_cap}")
            
            # Show dual-layer risk system status
            if self.risk_manager:
                x, y = int(probe.position.x), int(probe.position.y)
                if 0 <= x < self.risk_manager.map_width and 0 <= y < self.risk_manager.map_height:
                    live_risk = self.risk_manager.live_risk_map[x, y]
                    learned_risk = self.risk_manager.learned_risk_map[x, y]
                    combined_risk = self.risk_manager.get_combined_risk(x, y)
                    exploration_count = self.risk_manager.exploration_map[x, y]
                    print(f"[RISK] Current position - Live: {live_risk:.2f}, Learned: {learned_risk:.2f}, Combined: {combined_risk:.2f}, Visits: {exploration_count}")
                    print(f"[RISK] Total deaths logged: {self.failure_logger.stats['total_deaths']}")
            
        # Track path history for visualization and failure analysis
        if len(self.path) == 0 or probe.position.distance_to(self.path[-1]) > 1.0:
            self.path.append(probe.position)
            # Add to path history for failure context (limited size)
            self.path_history.append(probe.position)
            if len(self.path_history) > self.max_path_history:
                self.path_history.pop(0)
            # Limit main path history length for visualization
            if len(self.path) > 100:
                self.path.pop(0)
            
            # Add success reward to risk manager for successful movement
            if self.risk_manager:
                self.risk_manager.add_success_reward(probe.position)

        # Get current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Simple movement towards waypoint
        waypoint_reached = probe.position.distance_to(current_waypoint) < 0.1  # 0.1 is close enough to consider waypoint reached
        if not waypoint_reached:  # If not at waypoint
            # Check if waypoint is currently dangerous
            waypoint_risk = 0.0
            if self.risk_manager:
                # Check live enemy risk at waypoint
                for enemy in self.enemy_units:
                    distance = enemy.position.distance_to(current_waypoint)
                    if distance <= config.ENEMY_PRESENCE_RADIUS:
                        waypoint_risk += config.ENEMY_PRESENCE_PENALTY * (1.0 - distance / config.ENEMY_PRESENCE_RADIUS)
                
                # Check learned risk at waypoint
                wx, wy = int(current_waypoint.x), int(current_waypoint.y)
                if (0 <= wx < self.risk_manager.map_width and 0 <= wy < self.risk_manager.map_height):
                    learned_risk = self.risk_manager.learned_risk_map[wx, wy]
                    waypoint_risk += learned_risk * config.LEARNED_RISK_WEIGHT
            
            # If waypoint is too dangerous, find a safer intermediate waypoint
            if waypoint_risk > config.PATHFINDING_RISK_THRESHOLD * 3:
                print(f"[RISK] Waypoint too dangerous (risk={waypoint_risk:.2f}), finding safer route")
                # Find a safer position between probe and waypoint
                direction = (current_waypoint - probe.position).normalized
                for step_distance in [3.0, 5.0, 8.0]:  # Try different distances
                    safe_pos = probe.position + (direction * step_distance)
                    safe_risk = 0.0
                    
                    # Check if this intermediate position is safer
                    for enemy in self.enemy_units:
                        enemy_dist = enemy.position.distance_to(safe_pos)
                        if enemy_dist <= config.ENEMY_PRESENCE_RADIUS:
                            safe_risk += config.ENEMY_PRESENCE_PENALTY * (1.0 - enemy_dist / config.ENEMY_PRESENCE_RADIUS)
                    
                    if safe_risk < waypoint_risk * 0.5:  # Much safer
                        current_waypoint = safe_pos
                        print(f"[RISK] Using safer intermediate waypoint at {safe_pos} (risk={safe_risk:.2f})")
                        break
            
            # Always try pathfinding first
            try:
                # Start with map_analyzer's base grid
                cost_grid = self.map_data.get_pyastar_grid().astype(np.float32)
                
                # Add enemy presence costs (optimized)
                for enemy in self.enemy_units:
                    enemy_pos = enemy.position
                    x_center, y_center = int(enemy_pos.x), int(enemy_pos.y)
                    
                    # Quick bounds check
                    if not (config.ENEMY_PRESENCE_RADIUS <= x_center < cost_grid.shape[0] - config.ENEMY_PRESENCE_RADIUS and
                            config.ENEMY_PRESENCE_RADIUS <= y_center < cost_grid.shape[1] - config.ENEMY_PRESENCE_RADIUS):
                        continue
                    
                    # Optimized: only check in smaller radius and use pre-computed distances
                    radius_sq = config.ENEMY_PRESENCE_RADIUS * config.ENEMY_PRESENCE_RADIUS
                    for dx in range(-config.ENEMY_PRESENCE_RADIUS, config.ENEMY_PRESENCE_RADIUS + 1):
                        for dy in range(-config.ENEMY_PRESENCE_RADIUS, config.ENEMY_PRESENCE_RADIUS + 1):
                            dist_sq = dx*dx + dy*dy
                            if dist_sq <= radius_sq:
                                x_grid, y_grid = x_center + dx, y_center + dy
                                if cost_grid[x_grid, y_grid] != np.inf:
                                    falloff = max(0, 1 - (math.sqrt(dist_sq) / config.ENEMY_PRESENCE_RADIUS))
                                    cost_grid[x_grid, y_grid] += config.ENEMY_PRESENCE_PENALTY * falloff
                
                # Add learned risk costs (optimized - only near current path)
                if self.risk_manager and hasattr(self, 'current_path') and len(self.current_path) > 1:
                    # Only apply learned risk near the current path for performance
                    path_radius = 15  # Check learned risk within 15 units of path
                    for path_point in self.current_path[::3]:  # Sample every 3rd point
                        px, py = int(path_point.x), int(path_point.y)
                        
                        # Check learned risk in small area around path
                        for dx in range(-path_radius, path_radius + 1, 2):  # Step by 2 for performance
                            for dy in range(-path_radius, path_radius + 1, 2):
                                x_grid, y_grid = px + dx, py + dy
                                if (0 <= x_grid < min(self.risk_manager.map_width, cost_grid.shape[0]) and
                                    0 <= y_grid < min(self.risk_manager.map_height, cost_grid.shape[1])):
                                    learned_risk = self.risk_manager.learned_risk_map[x_grid, y_grid]
                                    if learned_risk > config.PATHFINDING_RISK_THRESHOLD:
                                        if x_grid < cost_grid.shape[0] and y_grid < cost_grid.shape[1] and cost_grid[x_grid, y_grid] != np.inf:
                                            risk_cost = min(config.PATHFINDING_RISK_CAP, 
                                                          learned_risk * config.LEARNED_RISK_WEIGHT)
                                            cost_grid[x_grid, y_grid] += risk_cost
                
                # Convert positions to Point2 for map_analyzer
                start_pos = Point2((probe.position.x, probe.position.y))
                target_pos = Point2((current_waypoint.x, current_waypoint.y))
                
                # Use map_analyzer's pathfinding WITH the modified grid
                path = self.map_data.pathfind(
                    start=start_pos,
                    goal=target_pos,
                    grid=cost_grid,  # Apply the modified grid!
                    smoothing=True
                )
                
                if path and len(path) > 1:
                    # Move to the next point in the path
                    next_point = path[1] if len(path) > 1 else path[0]
                    probe.move(next_point)
                    # Store the current path for visualization
                    self.current_path = path
                    
                    # Debug output for pathfinding
                    if self.loop_count % 10 == 0:  # Don't spam the console
                        print(f"[PATH] Moving to point {next_point} on path to waypoint {current_waypoint}")
                        print(f"[PATH] Distance to waypoint: {probe.position.distance_to(current_waypoint):.2f}")
                else:
                    print(f"[PATH] No path found to {current_waypoint}, falling back to direct movement")
                    
                    # Log pathfinding failure with context
                    context = {
                        "current_waypoint_index": self.current_waypoint_index,
                        "start_position": {"x": start_pos.x, "y": start_pos.y},
                        "goal_position": {"x": target_pos.x, "y": target_pos.y},
                        "path_history": self.path_history.copy(),
                        "enemy_units": [
                            {
                                "type": str(unit.type_id),
                                "position": {"x": unit.position.x, "y": unit.position.y},
                                "distance": unit.distance_to(start_pos)
                            }
                            for unit in self.enemy_units if unit.distance_to(start_pos) < 15
                        ],
                        "game_time": self.time,
                        "loop": self.state.game_loop,
                        "risk_threshold": config.PATHFINDING_RISK_THRESHOLD,
                        "max_risk_value": float(self.risk_map.max()) if self.risk_map.size > 0 else 0.0
                    }
                    
                    # Log the failure with enhanced context
                    self.failure_logger.log_pathfinding_failure(start_pos, target_pos, context)
                    
                    # Fall back to direct movement
                    probe.move(current_waypoint)
                    self.current_path = [probe.position, current_waypoint]  # Show direct path in debug
                    
            except Exception as e:
                print(f"[PATH] Error in pathfinding: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                # Only fall back to direct movement on error
                probe.move(current_waypoint)
        else:
            # Reached the waypoint, move to next
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            print(f"[MOVE] Reached waypoint, moving to next: {self.waypoints[self.current_waypoint_index]}")
        
        # Lightweight debug output (text only - no expensive drawing)
        if self.show_debug and iteration % 50 == 0:  # Reduced frequency
            try:
                current_wp = self.waypoints[self.current_waypoint_index]
                print(f"[DEBUG] WP {self.current_waypoint_index}: {current_wp}, {len(self.enemy_units)} enemies")
            except Exception:
                pass

    def update_risk_map(self):
        """Update the risk map based on enemy positions, death log, and failure statistics for adaptive learning"""
        if self.risk_map is None or self.risk_map.size == 0:
            return
        
        # Update the new dual-layer risk management system
        if self.risk_manager:
            # Apply temporal decay to live risk map
            self.risk_manager.apply_temporal_decay()
            
            # Add small live risk around enemy units (downsampled to reduce noise)
            if self.state.game_loop % 1 == 0:
                for unit in self.enemy_units:
                    self.risk_manager.add_enemy_presence_risk(unit.position)
            
            # Update the legacy risk map with combined risk for backward compatibility
            # Convert RiskManager's coordinate system to match map_analyzer's grid dimensions
            if hasattr(self, 'risk_map') and self.risk_map is not None:
                # Ensure risk_map matches the current map_analyzer grid dimensions
                if self.risk_map.shape != (self.risk_manager.map_width, self.risk_manager.map_height):
                    self.risk_map = np.zeros((self.risk_manager.map_width, self.risk_manager.map_height), dtype=np.float32)
                
                # Update risk_map with combined risk values
                for x in range(self.risk_manager.map_width):
                    for y in range(self.risk_manager.map_height):
                        self.risk_map[x, y] = self.risk_manager.get_combined_risk(x, y)
            return
            
        # Legacy risk map update (fallback if RiskManager not available)
        # Decay existing risk - simulates risk fading over time if no new deaths/failures occur there
        self.risk_map = self.risk_map * 0.95
        
        # Add risk around enemy units
        for unit in self.enemy_units:
            pos = unit.position.rounded
            x, y = int(pos.x), int(pos.y)
            if 0 <= x < self.risk_map.shape[0] and 0 <= y < self.risk_map.shape[1]:
                # Higher risk closer to enemy units
                self.risk_map[x, y] = max(self.risk_map[x, y], 10)
                
                # Add decreasing risk in radius
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.risk_map.shape[0] and 
                            0 <= ny < self.risk_map.shape[1]):
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance <= 5:  # Only affect nearby tiles
                                risk = max(0, 10 - distance)  # Linear falloff
                                self.risk_map[nx, ny] = max(self.risk_map[nx, ny], risk)
        
        # Add risk from death log (persistent high risk areas)
        for death_pos in self.death_log:
            x, y = int(death_pos.x), int(death_pos.y)
            if 0 <= x < self.risk_map.shape[0] and 0 <= y < self.risk_map.shape[1]:
                # Add high risk at death location with larger radius
                self.risk_map[x, y] = max(self.risk_map[x, y], 30)  # Very high risk at death location
                
                # Add decreasing risk in larger radius around death location
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.risk_map.shape[0] and 
                            0 <= ny < self.risk_map.shape[1]):
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance <= 10:  # Larger radius for death areas
                                # Higher risk closer to death location
                                risk = max(0, 20 - (distance * 2))  # Linear falloff
                                self.risk_map[nx, ny] = max(self.risk_map[nx, ny], risk)
        
        # Add high-risk areas from failure logger for adaptive learning
        high_risk_areas = self.failure_logger.get_high_risk_areas()
        for area in high_risk_areas:
            x, y = int(area.x), int(area.y)
            if 0 <= x < self.risk_map.shape[0] and 0 <= y < self.risk_map.shape[1]:
                # Very high risk at areas with repeated failures
                self.risk_map[x, y] = max(self.risk_map[x, y], 40)  # Even higher than death locations
                
                # Add decreasing risk in larger radius around high risk areas
                for dx in range(-12, 13):
                    for dy in range(-12, 13):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.risk_map.shape[0] and 
                            0 <= ny < self.risk_map.shape[1]):
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance <= 12:  # Larger radius for adaptive learning areas
                                # Higher risk closer to high-risk area
                                risk = max(0, 25 - (distance * 2))  # Steeper falloff
                                self.risk_map[nx, ny] = max(self.risk_map[nx, ny], risk)
        
        # Add risk around enemy units
        for unit in self.enemy_units:
            pos = unit.position.rounded
            x, y = int(pos.x), int(pos.y)
            if 0 <= x < self.risk_map.shape[0] and 0 <= y < self.risk_map.shape[1]:
                # Higher risk closer to enemy units
                self.risk_map[x, y] = max(self.risk_map[x, y], 10)
                
                # Add decreasing risk in radius
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.risk_map.shape[0] and 
                            0 <= ny < self.risk_map.shape[1]):
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance <= 5:  # Only affect nearby tiles
                                risk = max(0, 10 - distance)  # Linear falloff
                                self.risk_map[nx, ny] = max(self.risk_map[nx, ny], risk)
        
        # Add risk from death log (persistent high risk areas)
        for death_pos in self.death_log:
            x, y = int(death_pos.x), int(death_pos.y)
            if 0 <= x < self.risk_map.shape[0] and 0 <= y < self.risk_map.shape[1]:
                # Add high risk at death location with larger radius
                self.risk_map[x, y] = max(self.risk_map[x, y], 30)  # Very high risk at death location
                
                # Add decreasing risk in larger radius around death location
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.risk_map.shape[0] and 
                            0 <= ny < self.risk_map.shape[1]):
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance <= 10:  # Larger radius for death areas
                                # Higher risk closer to death location
                                risk = max(0, 20 - (distance * 2))  # Linear falloff
                                self.risk_map[nx, ny] = max(self.risk_map[nx, ny], risk)
                                
    def handle_probe_death(self):
        """Handle probe death - log death, record position, reset path for respawn"""
        death_pos = None
        
        # Get the death position from the last known path point
        if self.path:
            death_pos = self.path[-1]
        elif hasattr(self, 'last_probe_position'):
            death_pos = self.last_probe_position
            
        if not death_pos:
            print("[DEATH] No position data available for probe death")
            return
            
        print(f"[DEATH] Probe died at {death_pos}, logging for adaptive learning")
        
        # Clear the probe tag as it's now invalid
        self.probe_tag = None
        
        # Prepare context data for the log
        context = {
            "current_waypoint": self.waypoints[self.current_waypoint_index],
            "path_history": self.path_history,
            "nearby_enemies": [],
            "game_time": self.time,
            "game_loop": self.state.game_loop,
            "death_count": self.failure_logger.stats["total_deaths"] + 1,
            "visible": self.is_visible(death_pos)
        }
        
        # Add nearby enemy units to context
        for enemy in self.enemy_units:
            if enemy.position.distance_to(death_pos) < 15:
                context["nearby_enemies"].append({
                    "type": enemy.type_id.name,
                    "position": {"x": enemy.position.x, "y": enemy.position.y},
                    "distance": enemy.position.distance_to(death_pos)
                })
        
        # Log the death to the failure logger
        self.failure_logger.log_probe_death(death_pos, context)
        
        # Add death risk to the new dual-layer risk management system
        if self.risk_manager:
            self.risk_manager.add_death_risk(death_pos, is_current_session=True)
            print(f"[RISK] Added death risk to dual-layer system at {death_pos}")
        
        # Reset path tracking for the new probe
        self.path = []
        self.path_history = []
        
        # Reset waypoint navigation so the new probe starts from the beginning
        self.current_waypoint_index = 0
        self.waypoint_reached = False
        print(f"[NAVIGATION] Waypoint progress reset to beginning (waypoint {self.current_waypoint_index})")
        
        # Update risk map with death location (keep for backward compatibility)
        self.update_risk_map()
        
        # Save logs immediately
        try:
            self.failure_logger.save_to_file()
            print(f"[LOG] Death #{context['death_count']} logged and saved at position {death_pos}")
        except Exception as e:
            print(f"[ERROR] Failed to save death log: {e}")
        
        # Try to get the probe again (it might have respawned)
        probe = self.units.by_tag(self.probe_tag) if self.probe_tag else None
        if not probe and self.units:
            # If we have units but no probe with the stored tag, get the first worker
            worker = self.workers.first
            if worker:
                self.probe_tag = worker.tag

    def get_next_position(self,unit: Unit) -> Point2:
        try:
            self.unvisited_positions.remove(unit.position.rounded)
        except ValueError:
            pass
       

        closest_point: Point2 = None
        closest_distance: int = 9999

        for unvisited in self.unvisited_positions:
            influence = self.influence_grid[unvisited.x][unvisited.y]
            if influence == 10:
                continue
            distance = unit.position.distance_to(unvisited)
            if distance < closest_distance:
                closest_point = unvisited
                closest_distance = distance
            if distance == 1:
                break
        return closest_point


    def _draw_path_box(self, p, color):
        h = self.get_terrain_z_height(p)
        pos = Point3((p.x, p.y, h))
        box_r = 1
        p0 = Point3((pos.x - box_r, pos.y - box_r, pos.z + box_r)) + Point2((0.5, 0.5))
        p1 = Point3((pos.x + box_r, pos.y + box_r, pos.z - box_r)) + Point2((0.5, 0.5))
        self.client.debug_box_out(p0, p1, color=color)

    def _draw_point_list(self, points, color):
        if len(points) > 50:
            points_to_draw = points[-50:]
        else:
            points_to_draw = points
        for point in points_to_draw:
            self._draw_path_box(point, color)

    async def on_end(self, game_result: Result):
        """Handle end of game - save logs and report path efficiency"""
        print(f"[END] Game ended with result: {game_result}")
        
        # Save persistent risk data for future sessions
        if self.risk_manager:
            self.risk_manager.save_persistent_data()
            print(f"[RISK] Saved persistent learning data for future sessions")
        
        # Save logs at end of game
        try:
            self.failure_logger.save_to_file()
            print(f"[LOG] Final game logs saved")
        except Exception as e:
            print(f"[ERROR] Failed to save logs at game end: {e}")
            
        # Print path efficiency report
        if hasattr(self, 'path') and len(self.path) > 1:
            self.print_path_efficiency()

    def print_path_efficiency(self):
        if len(self.path) < 2:
            print("Not enough path points to calculate efficiency")
            return
            
        path_length = sum(self.map_data.distance(self.path[i], self.path[i+1]) for i in range(len(self.path)-1))
        if path_length > 0:
            direct_distance = self.map_data.distance(self.path[0], self.path[-1])
            efficiency = direct_distance / path_length if path_length > 0 else 0
            print(f"Path Analysis:")
            print(f"- Start: {self.path[0]}")
            print(f"- End: {self.path[-1]}")
            print(f"- Direct Distance: {direct_distance:.2f}")
            print(f"- Total Path Length: {path_length:.2f}")
            print(f"- Path Efficiency: {efficiency:.3f} (1.0 = perfect)")
            
            # Calculate and display additional metrics
            num_segments = len(self.path) - 1
            avg_segment_length = path_length / num_segments if num_segments > 0 else 0
            print(f"- Number of Segments: {num_segments}")
            print(f"- Average Segment Length: {avg_segment_length:.2f}")
