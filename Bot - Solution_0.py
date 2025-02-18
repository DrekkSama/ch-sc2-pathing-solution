from sc2.bot_ai import BotAI, Race
from sc2.data import Result

from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId

import numpy as np
import random

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
        self.target = None
        self.probe_tag = None
        self.path_length = 0
        self.direct_distance = 0
        self.visited_positions = {}
        self.unvisited_positions = []
        self.last_scout_position: Point2 = None
        self.scout_destination: Point2 = None
        self.secret_location: Point2 = None

    async def on_start(self):
        self.map_data = MapData(self, loglevel="DEBUG", arcade=True)

        grid: np.ndarray = self.map_data.get_pyastar_grid()
        print(f"grid length: {len(grid)}")

        if self.map_data:
            self.probe_tag = self.workers.first.tag  # Get the first worker (probe)
            self.p0 = self.workers.first.position  # Starting position of the probe
            
        # pathing_grid = self.game_info.pathing_grid
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                coords = (x, y)
                point = Point2(coords)
                if grid[point] != np.inf:
                    self.unvisited_positions.append(point)
        
        self.create_random_secret_location(grid)

    async def on_step(self, iteration: int):
        # check the probe and update the path
        probe = self.units.by_tag(self.probe_tag)
        if probe and probe.position not in self.path:
            self.path.append(probe.position)

        self._draw_point_list(self.path, color=RED)
        self._draw_path_box(probe.position, color=GREEN)

        if self.last_scout_position and self.last_scout_position.distance_to(probe) < 0.001:
            self.unvisited_positions.remove(self.scout_destination)

        self.scout_destination = self.get_next_position(probe)
        # TODO: Implement movement logic after here
        if self.scout_destination:
            probe.move(self.scout_destination)

        self.last_scout_position = probe.position

    def get_next_position(self,unit: Unit) -> Point2:
        # remove_current_position_from_unvisited
        try:
            self.unvisited_positions.remove(unit.position.rounded)
        except ValueError:
            pass

        # find nearest unvisited
        closest_point: Point2 = None
        closest_distance: int = 9999

        for unvisited in self.unvisited_positions:
            distance = unit.position.distance_to(unvisited)
            if distance < closest_distance:
                closest_point = unvisited
                closest_distance = distance
            if distance == 1:
                break
        return closest_point

    def create_random_secret_location(self, grid: np.ndarray):
        grid_width = grid.shape[0]
        grid_height = grid.shape[1]

        random_x = random.randint(0, grid_width)
        random_y = random.randint(0, grid_height)
        coords = (random_x, random_y)
        self.secret_location = Point2(coords)
        while grid[self.secret_location] == np.inf:
            random_x = random.randint(0, grid_width)
            random_y = random.randint(0, grid_height)
            coords = (random_x, random_y)
            self.secret_location = Point2(coords)
        # self.client.debug_create_unit([UnitTypeId.MOTHERSHIP, 1, self.game_info.map_center, 1])
        print(f"random location {self.secret_location}")

    def found_secret_location(self):
        return self.is_visible(self.secret_location)

    def _draw_path_box(self, p, color):
        """ Draws a debug box at a given position to visualize the path. """
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

        self._draw_path_box(self.secret_location, BLUE)

    async def on_end(self, game_result: Result):
        self.p1 = self.workers.first.position
        self.print_path_efficiency()

    def print_path_efficiency(self):
        path_length = sum(self.map_data.distance(self.path[i], self.path[i+1]) for i in range(len(self.path)-1))
        if len(self.path) > 1:
            direct_distance = self.map_data.distance(self.path[0], self.path[-1])
            efficiency = direct_distance / path_length if path_length > 0 else 0
            print(f"Path Taken: {self.path}")
            print(f"Direct Distance: {direct_distance}")
            print(f"Total Path Length: {path_length}")
            print(f"Path Efficiency: {efficiency:.3f} (1.0 = perfect)")
