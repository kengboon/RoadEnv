import numpy as np
from road_env import utils
from road_env.envs.common.action import action_factory
from road_env.road.road import Road
from road_env.utils import Vector
from road_env.vehicle.kinematics import Vehicle

class Pedestrian(Vehicle):
    LENGTH = 1.5
    WIDTH = 2

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering',
                 start_lane: int = 0,
                 max_lane: int = 4,
                 ego_vehicle: Vehicle = None,
                 env = None,
                 action_config: dict = None,
                 crossing_config: dict = None
                 ):
        super().__init__(road, position, heading, speed, predition_type)
        self.check_collisions = False
        self.ego_vehicle = ego_vehicle
        self.start_lane =start_lane
        self.max_lane = max_lane
        self.config = crossing_config
        self.action_type = action_factory(env, action_config)
        self.action_type.controlled_vehicle = self
        self.action_space = self.action_type.space()
        self.last_steering = 0
        self.last_acceleration = 0
        self.target_heading = self.heading
        self.crossing = False
        self.crossed = False

    def act(self, action: dict | str = None) -> None:
        if action:
            # Called from ActionType.act()
            # Forward action to vehicle (step later)
            super().act(action=action)
        else:
            # Called from Road.act()
            acceleration, steering = self.action_space.sample()

            # Check have crossed the road
            if self.crossing and self.start_lane != self.lane_index[2] and not self.on_road:
                self.crossing = False
                self.crossed = True
                self.target_heading = self.road.np_random.choice((0, np.pi))

            # Evaluate to start crossing the road
            if self.ego_vehicle and not self.crossing and not self.crossed:
                # Check ego vehicle distance within range
                if self.config["min_distance"] <= self.distance_to_ego <= self.config["max_distance"]:
                    # Probability of crossing road
                    self.crossing = self.road.np_random.random() < self.config["probability"]
                    self.crossed = False

            # Steer angle to cross the road
            if self.crossing:
                if self.start_lane == 0:
                    self.target_heading = np.pi / 2
                else:
                    self.target_heading = -np.pi / 2
                acceleration = 1 # Max acceleration

            # Get steering toward target_heading
            steering = self.get_steering(self.target_heading, normalize=True)

            self.last_steering = steering
            self.last_acceleration = acceleration
            self.action_type.act([acceleration, steering])

    def get_steering(self, target_heading: float, normalize: bool = True):
        steering = target_heading - self.heading
        if steering > np.pi:
            steering -= 2 * np.pi
        elif steering < -np.pi:
            steering += 2 * np.pi
        if normalize:
            steering = utils.lmap(steering, self.action_type.steering_range, [-1, 1])
        return steering

    @property
    def distance_to_ego(self):
        if self.ego_vehicle:
            return np.linalg.norm(self.position - self.ego_vehicle.position) - self.WIDTH / 2

    def to_dict(self, origin_vehicle: Vehicle = None, observe_intentions: bool = True) -> dict:
        d = super().to_dict(origin_vehicle, observe_intentions)
        d['class'] = 1 if self.on_road else 0.5
        return d