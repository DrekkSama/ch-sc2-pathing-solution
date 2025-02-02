from sc2.bot_ai import BotAI, Race
from sc2.data import Result

from sc2.units import Units
from sc2.ids.unit_typeid import UnitTypeId

from sc2.position import Point2, Point3
from map_analyzer import MapData


GREEN = Point3((0, 255, 0))
RED = Point3((255, 0, 0))


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

    async def on_start(self):
        self.map_data = MapData(self, loglevel="DEBUG", arcade=True)

        if self.map_data:
            self.probe_tag = self.workers.first.tag  # Get the first worker (probe)
            self.p0 = self.workers.first.position  # Starting position of the probe
            
        # TODO: Implement pathfinding logic after here

    async def on_step(self, iteration: int):
        # check the probe and update the path
        probe = self.units.by_tag(self.probe_tag)
        if probe and probe.position not in self.path:
            self.path.append(probe.position)

        self._draw_point_list(self.path, color=RED)
        self._draw_path_box(probe.position, color=GREEN)

        # TODO: Implement movement logic after here

    
    
    
    def _draw_path_box(self, p, color):
        """ Draws a debug box at a given position to visualize the path. """
        h = self.get_terrain_z_height(p)
        pos = Point3((p.x, p.y, h))
        box_r = 1
        p0 = Point3((pos.x - box_r, pos.y - box_r, pos.z + box_r)) + Point2((0.5, 0.5))
        p1 = Point3((pos.x + box_r, pos.y + box_r, pos.z - box_r)) + Point2((0.5, 0.5))
        self.client.debug_box_out(p0, p1, color=color)

    def _draw_point_list(self, points, color):
        for point in points:
            self._draw_path_box(point, color)

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
