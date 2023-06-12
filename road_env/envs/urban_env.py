from typing import Dict, Text
import numpy as np
from road_env import utils
from road_env.envs.common.abstract import AbstractEnv
from road_env.envs.common.action import Action
from road_env.road.graphics import RoadObjectGraphics
from road_env.road.road import Road, RoadNetwork
from road_env.vehicle.graphics import VehicleGraphics
from road_env.vehicle.human import Pedestrian
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
                "type": "LidarKinematicsObservation",
                "features": ["class", "x", "y", "vx", "vy", "heading"],
                "cells": 16,
                "maximum_range": 60,
                "see_behind": False,
                "display_line": True,
                "display_unobserved": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-6.944, 2.778], # -25 km/h/s - 10 km/h/s
                "steering_range": [-0.524, 0.524], # 30 degree in radian
                "speed_range": [0, 22.222] # 80 km/h
            },
            "random_seed": 42,
            "lanes_count": 4,
            "road_length": 1000,
            "speed_limit": 16.667, # 60 km/h
            "initial_lane_id": 1,
            "obstacle_count": 80,
            "obstacle_size": 1.5,
            "pedestrians": {
                "count": 20,
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "acceleration_range": [-0.833, 0.5], # -3 km/h/s - 
                    "steering_range": [-0.524, 0.524], # 30 degree in radian
                    "speed_range": [0, 0.833] # 3 km/h
                }
            },
            "duration": 999,  # [s]
            "collision_reward": -1,
            "on_lane_reward": 0.2,
            "on_road_reward": 0.1,
            "speed_reward": 0.3,
            "reward_speed_range": [13.889, 16.667, 19.167], # 50-60-69 km/h
            "normalize_reward": True,
            "offroad_terminal": True,
            "show_trajectories": False
        })
        return config

    def _reset(self) -> None:
        seed_seq = np.random.SeedSequence(self.config["random_seed"])
        self._generator = np.random.Generator(np.random.PCG64(seed_seq))
        self._make_road()
        self._make_vehicles()
        self._make_obstacles()
        self._make_pedestrians()

    def _make_road(self) -> None:
        road_network = RoadNetwork.straight_road_network(
            lanes=self.config["lanes_count"],
            length=self.config["road_length"],
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
            self.road.network.get_lane(("0", "1", self.config["initial_lane_id"])).position(200, 0),
        )
        ego_vehicle.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.vehicle_lane = self.config["initial_lane_id"]

    def _make_obstacles(self) -> None:
        for _ in range(self.config["obstacle_count"]):
            lane = self.road.network.get_lane(("0", "1", self._generator.choice((0, 3))))
            pos_x = self._generator.integers(5, lane.end[0])
            obstacle = Obstacle(
                self.road,
                lane.position(pos_x, 0),
            )
            width = self._generator.random() + self.config["obstacle_size"]
            length = width + self._generator.random() * width
            obstacle.change_size(length, width)
            obstacle.color = RoadObjectGraphics.BLUE
            self.road.objects.append(obstacle)

    def _make_pedestrians(self) -> None:
        for _ in range(self.config["pedestrians"]["count"]):
            lane_index = self._generator.integers(0, self.config["lanes_count"]-1)
            lane = self.road.network.get_lane(("0", "1", lane_index))
            pos_x = self._generator.integers(5, lane.end[0])
            if lane_index <= self.config["lanes_count"] / 2:
                heading = 1.05 + self._generator.random() * 1.05
            else:
                heading = -1.05 - self._generator.random() * 1.05
            target_lane = self.config["lanes_count"]-1
            pedestrian = Pedestrian(
                self.road,
                position=lane.position(pos_x, 0),
                speed=3,
                heading=heading)
            pedestrian.configure(self, self.config["pedestrians"])
            pedestrian.target_lane = target_lane
            pedestrian.others.append(self.vehicle)
            self.road.pedestrians.append(pedestrian)
            self.road.vehicles.append(pedestrian)
        self.pedestrians = self.road.pedestrians

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["on_road_reward"] + self.config["on_lane_reward"] + self.config["speed_reward"]],
                                 [0, 1])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        rewards = {
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
            "on_lane_reward": 0,
            "speed_reward": 0
        }
        # On-lane reward if no collision
        if not self.vehicle.crashed:
            if self.vehicle_lane == self.vehicle.lane_index[2]:
                rewards["on_lane_reward"] = 1
            elif self.vehicle.lane_index[2] not in (0, self.config["lanes_count"]-1):
                rewards["on_lane_reward"] = 0.5
            self.vehicle_lane = self.vehicle.lane_index[2]

        # Speed reward if no collision
        if not self.vehicle.crashed:
            low_speed, desired_speed, high_speed = self.config["reward_speed_range"]
            # Use forward speed, see https://github.com/Farama-Foundation/HighwayEnv/issues/268
            forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            if low_speed <= forward_speed <= high_speed:
                speed_tolerance = max(abs(desired_speed - low_speed), abs(desired_speed - high_speed))
                speed_diff = abs(forward_speed - desired_speed)
                rewards["speed_reward"] = speed_diff / speed_tolerance
        return rewards

    def _is_terminated(self) -> bool:
        return (self.vehicle.crashed or
            self.config["offroad_terminal"] and not self.vehicle.on_road)
    
    def _is_truncated(self) -> bool:
        return self.time >= self.config['duration']