from road_env.envs.common.action import action_factory
from road_env.road.road import Road
from road_env.utils import Vector
from road_env.vehicle.kinematics import Vehicle
from road_env.vehicle.objects import RoadObject, Obstacle

class Pedestrian(Vehicle):
    LENGTH = 1.5
    WIDTH = 2

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed, predition_type)
        self.check_collisions = False

    def configure(self, env, config):
        self.env = env
        self.config = config
        self.action_type = action_factory(env, config["action"])
        self.action_space = self.action_type.space()

    def handle_collisions(self, other: RoadObject, dt: float = 0) -> None:
        return

    def act(self, action: dict | str = None) -> None:
        action = self.action_space.sample()
        return super().act({
            "acceleration": action[0],
            "steering": action[1],
        })