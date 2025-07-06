from sc2.bot_ai import BotAI, Race
from sc2.data import Result

from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId

import numpy as np
import random
import math

from sc2.position import Point2, Point3
from map_analyzer import MapData


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
        
        # Risk management
        self.death_log = []
        self.enemy_units = []
        self.risk_map = None
        
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
        
        # Initialize risk map with zeros (same shape as grid)
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
        self.waypoints = [
            Point2((35.27, 140.06)),  # Area 1
            Point2((47.20, 108.33)),  # Area 2
            Point2((107.07, 112.04)), # Area 3
            Point2((137.66, 136.96)), # Area 4
            Point2((134.84, 69.69)),  # Area 5
            Point2((84.90, 109.94))   # Area 6
        ]
        
        # Add a small offset to the first waypoint to ensure it's reachable
        # This helps prevent the probe from getting stuck on the initial position
        self.waypoints[0] = self.waypoints[0].offset(Point2((1.0, 1.0)))
        
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
        else:
            print("[ERROR] No workers found!")
            return
        
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
        # Get the probe
        probe = self.units.by_tag(self.probe_tag) if self.probe_tag else None
        
        # Check if probe died and handle respawn
        if self.probe_tag and not probe and self.workers:
            # Probe died, handle death and get new probe
            self.handle_probe_death()
            probe = self.workers.first
            if probe:
                self.probe_tag = probe.tag
                print(f"[INFO] Probe respawned at {probe.position}")
        elif not probe and self.workers:
            # No probe assigned yet, get the first worker
            probe = self.workers.first
            if probe:
                self.probe_tag = probe.tag
                print(f"[INFO] Assigned new probe: {probe.tag} at {probe.position}")
        
        if not probe:
            print("[WARNING] No probe found to control")
            return
            
        # Update risk map with current enemy positions
        self.enemy_units = self.enemy_units | self.all_enemy_units
        self.update_risk_map()
        
        # Print debug info periodically
        if iteration % 50 == 0:
            print(f"[DEBUG] Probe at {probe.position}, health: {probe.health}")
            print(f"[DEBUG] Current waypoint: {self.waypoints[self.current_waypoint_index]}")
            print(f"[DEBUG] Game state - Time: {self.time:.1f}, Supply: {self.supply_used}/{self.supply_cap}")
            
        # Track path history for visualization
        if len(self.path) == 0 or probe.position.distance_to(self.path[-1]) > 1.0:
            self.path.append(probe.position)
            if len(self.path) > 100:  # Limit path history length
                self.path.pop(0)

        # Get current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Simple movement towards waypoint
        waypoint_reached = probe.position.distance_to(current_waypoint) < 0.5  # 0.5 is close enough to consider waypoint reached
        if not waypoint_reached:  # If not at waypoint
            # Always try pathfinding first
            try:
                # Create a combined grid of influence and risk
                combined_grid = self.influence_grid.copy().astype(np.float32)
                
                # Add risk to the grid (higher risk = less desirable path)
                risk_threshold = 0.5
                risk_mask = (self.risk_map > risk_threshold)
                risk_values = np.minimum(10, 1 + (10 * self.risk_map[risk_mask]))
                combined_grid[risk_mask] = risk_values
                
                # Convert positions to Point2 if they aren't already
                start_pos = Point2((probe.position.x, probe.position.y))
                target_pos = Point2((current_waypoint.x, current_waypoint.y))
                
                # Find path using A*
                path = self.map_data.pathfind(
                    start=start_pos,
                    goal=target_pos,
                    grid=combined_grid,
                    smoothing=True,
                    sensitivity=1
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
        
        # Debug drawing
        if self.show_debug and iteration % 10 == 0:  # Don't draw every frame to save performance
            try:
                # Draw waypoints
                for i, wp in enumerate(self.waypoints):
                    if hasattr(self, 'get_terrain_height'):
                        height = self.get_terrain_height(Point2((wp.x, wp.y)))
                        pos_3d = Point3((wp.x, wp.y, height))
                        color = GREEN if i == self.current_waypoint_index else RED
                        self._draw_path_box(pos_3d, color=color)
                
                # Draw current position
                if hasattr(self, 'get_terrain_height'):
                    height = self.get_terrain_height(probe.position)
                    pos_3d = Point3((probe.position.x, probe.position.y, height))
                    self._draw_path_box(pos_3d, color=GREEN)
                
                # Draw line to current target
                target = self.waypoints[self.current_waypoint_index]
                if hasattr(self, 'get_terrain_height'):
                    start_height = self.get_terrain_height(probe.position)
                    end_height = self.get_terrain_height(target)
                    self.client.debug_line_out(
                        Point3((probe.position.x, probe.position.y, start_height)),
                        Point3((target.x, target.y, end_height)),
                        color=GREEN
                    )
                
                # Draw current path if it exists
                if hasattr(self, 'current_path') and len(self.current_path) > 1:
                    path_points = []
                    for point in self.current_path:
                        if hasattr(self, 'get_terrain_height'):
                            height = self.get_terrain_height(Point2((point.x, point.y)))
                            path_points.append(Point3((point.x, point.y, height + 0.5)))
                    
                    if len(path_points) > 1:
                        self.client.debug_polyline_out(path_points, color=BLUE)
                
            except Exception as e:
                print(f"[DEBUG] Error in debug drawing: {str(e)[:100]}")

    def handle_probe_death(self):
        """Handle probe death by recording the death location"""
        if self.path:
            death_pos = self.path[-1]  # Last known position before death
            self.death_log.append(death_pos)
            print(f"Probe died at {death_pos}")
        
        # Reset for respawn
        self.current_path = []
        self.path = []
        
        # Try to get the probe again (it might have respawned)
        probe = self.units.by_tag(self.probe_tag) if self.probe_tag else None
        if not probe and self.units:
            # If we have units but no probe with the stored tag, get the first worker
            worker = self.workers.first
            if worker:
                self.probe_tag = worker.tag
    
    def update_risk_map(self):
        """Update the risk map based on enemy positions and death log"""
        if self.risk_map is None or self.risk_map.size == 0:
            return
            
        # Decay existing risk
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
        if self.workers:
            self.p1 = self.workers.first.position
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
