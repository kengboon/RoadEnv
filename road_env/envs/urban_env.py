from typing import Dict, Text
from road_env.envs.common.abstract import AbstractEnv
from road_env.envs.common.action import Action
from road_env.road.road import Road, RoadNetwork
from road_env.vehicle.controller import ControlledVehicle
from road_env.vehicle.kinematics import Vehicle

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
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-6.94, 2.22], # -25 km/h/s, 8 km/h/s
                "steering_range": [-0.5236, 0.5236], # 30 degree in radian
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
    
    def _make_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(
            lanes=self.config["lanes_count"],
            speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _make_vehicles(self) -> None:
        pass

    def _rewards(self, action: Action) -> Dict[Text, float]:
        return {
            'collision_reward': float(self.vehicle.crashed),
            'on_road_reward': float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed
    
    def _is_truncated(self) -> bool:
        return self.time >= self.config['duration']