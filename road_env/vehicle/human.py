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
        self.target_lane = 0
        self.others = []

    def configure(self, env, config):
        self.env = env
        self.config = config
        self.action_type = action_factory(env, config["action"])
        self.action_space = self.action_type.space()

    def act(self, action: dict | str = None) -> None:
        acceleration, steering = self.action_space.sample()

        if any(abs(self.front_distance_to(other)) < self.LENGTH * self.WIDTH for other in self.others):
            acceleration = 0

        if self.lane_index[2] in (0, self.target_lane):
            # Scale up the steering action
            # thus increase chance of a u-turn
            if 0.25 <= steering < 0.5:
                steering += 0.5
            elif -0.5 > steering >= 0.25:
                steering -= 0.5

        return super().act({
            "acceleration": acceleration,
            "steering": steering,
        })

    def to_dict(self, origin_vehicle: Vehicle = None, observe_intentions: bool = True) -> dict:
        d = super().to_dict(origin_vehicle, observe_intentions)
        d['class'] = 1
        return d