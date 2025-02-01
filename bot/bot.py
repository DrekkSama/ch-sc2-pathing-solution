from sc2.bot_ai import BotAI, Race
from sc2.data import Result

from sc2.position import Point2, Point3
from map_analyzer import MapData

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
        self.map_data = MapData(self)
        self.probe_tag = self.workers[0].tag  # Get the first worker (probe)

        # Get start and goal locations (modify this as needed)
        reg_start = self.map_data.where_all(self.townhalls[0].position_tuple)[0]
        reg_end = self.map_data.where_all(self.enemy_start_locations[0].position)[0]

        self.p0 = reg_start.center
        self.p1 = reg_end.center

        # TODO: Implement pathfinding logic here

    async def on_step(self, iteration: int):
        probe = self.units.by_tag(self.probe_tag)
        if not probe:
            return  # If the probe is gone, exit early

        # Draw the probe’s current position in GREEN
        self._draw_path_box(probe.position, Point3((0, 255, 0)))

        # Draw the next step in the path in RED
        if self.path:
            self._draw_path_box(self.path[0], Point3((255, 0, 0)))

        # TODO: Implement movement logic here

    def _draw_path_box(self, p, color):
        """ Draws a debug box at a given position to visualize the path. """
        h = self.get_terrain_z_height(p)
        pos = Point3((p.x, p.y, h))
        box_r = 1
        p0 = Point3((pos.x - box_r, pos.y - box_r, pos.z + box_r)) + Point2((0.5, 0.5))
        p1 = Point3((pos.x + box_r, pos.y + box_r, pos.z - box_r)) + Point2((0.5, 0.5))
        self.client.debug_box_out(p0, p1, color=color)

    async def on_end(self, game_result):
        self.direct_distance = self.map_data.distance(self.p0, self.p1)
        self.path_length = sum(self.map_data.distance(self.path[i], self.path[i+1]) for i in range(len(self.path)-1))
        efficiency = self.direct_distance / self.path_length if self.path_length > 0 else 0
        print(f"Game Result: {game_result}")
        print(f"Direct Distance: {self.direct_distance}")
        print(f"Total Path Length: {self.path_length}")
        print(f"Path Efficiency: {efficiency:.3f} (1.0 = perfect)")

