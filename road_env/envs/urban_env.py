import random
from typing import Dict, Text
from road_env.envs.common.abstract import AbstractEnv
from road_env.envs.common.action import Action
from road_env.road.graphics import RoadObjectGraphics
from road_env.road.road import Road, RoadNetwork
from road_env.vehicle.graphics import VehicleGraphics
from road_env.vehicle.objects import Obstacle

class UrbanRoadEnv(AbstractEnv):
    '''
    A urban road environment.

    The vehicle is driving on a straight common road with two lanes, 
    obstacles may present at the side of road,
    pedestrians may walking on sidewalks and cross the road.
    Vehicle is rewarded for reaching speed close to limit, staying on
    designated lanes and avoiding collisions.
    '''

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy", "heading"]
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-6.94, 2.22], # -25 km/h/s, 8 km/h/s
                "steering_range": [-0.5236, 0.5236], # 30 degree in radian
            },
            "random_seed": 42,
            "lanes_count": 4,
            "speed_limit": 30,
            "initial_lane_id": 1,
            "obstacle_count": 900,
            "obstacle_size": 1.5,
            "duration": 40,  # [s]
            "collision_reward": -1,
            "high_speed_reward": 0.5,
            "on_road_reward": 0.1,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
            "show_trajectories": False
        })
        return config

    def _reset(self) -> None:
        random.seed(self.config["random_seed"])
        self._make_road()
        self._make_vehicles()
        self._make_obstacles()
        self._make_pedestrians()
        random.seed()
    
    def _make_road(self) -> None:
        road_network = RoadNetwork.straight_road_network(
            lanes=self.config["lanes_count"],
            speed_limit=self.config["speed_limit"]
        )
        self.road = Road(
            network=road_network,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _make_vehicles(self) -> None:
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            self.road.network.get_lane(("0", "1", self.config["initial_lane_id"])).position(5, 0),
        )
        ego_vehicle.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _make_obstacles(self) -> None:
        for _ in range(self.config["obstacle_count"]):
            lane = self.road.network.get_lane(("0", "1", random.choice((0, 3))))
            pos_x = random.randint(5, lane.end[0])
            obstacle = Obstacle(
                self.road,
                lane.position(pos_x, 0),
            )
            width = random.random() + self.config["obstacle_size"]
            length = width + random.random() * width
            obstacle.change_size(length, width)
            obstacle.color = RoadObjectGraphics.BLUE
            self.road.objects.append(obstacle)

    def _make_pedestrians(self) -> None:
        pass

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        return {
            'collision_reward': self.vehicle.crashed,
            'on_road_reward': self.vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        return (self.vehicle.crashed or
            self.config["offroad_terminal"] and not self.vehicle.on_road)
    
    def _is_truncated(self) -> bool:
        return self.time >= self.config['duration']