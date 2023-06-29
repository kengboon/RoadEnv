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
                "ego_features": ["y", "vx", "heading"],
                "features": ["class", "on_road", "distance"],
                "normalize": True,
                "cells": 16,
                "maximum_range": 60,
                "see_behind": False,
                "observe_angle": [-1.57, 1.57], # -90 deg - +90 deg
                "display_line": True,
                "display_unobserved": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-6.944, 2.778], # -25 km/h/s - 10 km/h/s
                "steering_range": [-0.524, 0.524], # 30 degree in radian
                "speed_range": [0, 19.167], # 69 km/h
                "shift_normalize": True,
            },
            "random_seed": 42,
            "obstacle_preset": None,
            "lanes_count": 4,
            "road_length": 500,
            "speed_limit": 16.667, # 60 km/h
            "initial_lane_id": 1,
            "obstacle_count": 30,
            "obstacle_size": 1.5,
            "pedestrians": {
                "count": 10,
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "acceleration_range": [-0.5, 0.5], # ~2 km/h/s 
                    "steering_range": [-np.pi / 2, np.pi / 2], # 90 degree in radian
                    "speed_range": [0, 0.833] # 3 km/h
                },
                "crossing": {
                    "min_distance": 5,
                    "max_distance": 50,
                    "probability": 0.333,
                }
            },
            "duration": 500,  # Time step
            "collision_reward": -1,
            "off_road_reward": -1,
            "off_lane_reward": -.75,
            "prolong_static_reward": -.75,
            "prolong_static_count": [20, 100],
            "low_speed_reward": -0.5,
            "low_speed_range": [0, 8.333], # 0-20 km/h
            "on_lane_reward": 0,
            "heading_reward": 0,
            "high_speed_reward": 1,
            "high_speed_range": [8.3333, 16.667, 19.167], # 20-60-69 km/h
            "normalize_reward": False,
            "reward_range": [-1, 1],
            "clip_reward": True,
            "offroad_terminal": True,
            "show_trajectories": False
        })
        return config

    def _reset(self) -> None:
        seed_seq = np.random.SeedSequence(self.config["random_seed"])
        self._generator = np.random.Generator(np.random.PCG64(seed_seq))
        self._preset_mode()
        self._make_road()
        self._make_vehicles()
        self._make_obstacles()
        self._make_pedestrians()

    def _preset_mode(self) -> None:
        match self.config["obstacle_preset"]:
            case 1: # Low occlusion
                self.configure({
                    "obstacle_count": 10,
                    "obstacle_size": 0.75
                })
            case 2: # Medium occlusion
                self.configure({
                    "obstacle_count": 25,
                    "obstacle_size": 1.5
                })
            case 3: # High occlusion
                self.configure({
                    "obstacle_count": 50,
                    "obstacle_size": 1.75
                })
            case 4: # Random level
                self.configure({
                    "obstacle_count": self._generator.integers(10, 50),
                    "obstacle_size": self._generator.random() + 0.75
                })

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
            self.road.network.get_lane(("0", "1", self.config["initial_lane_id"])).position(5, 0),
        )
        ego_vehicle.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.vehicle_lane = self.config["initial_lane_id"]
        self.static_counter = 0

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
            lane_index = self._generator.choice((0, self.config["lanes_count"] - 1))
            lane = self.road.network.get_lane(("0", "1", lane_index))
            pos_x = self._generator.integers(50, lane.end[0])
            pos_y = -lane.width if lane_index == 0 else lane.width
            heading = self._generator.choice((0, np.pi))
            pedestrian = Pedestrian(
                self.road,
                position=lane.position(pos_x, pos_y),
                heading=heading,
                speed=0,
                start_lane=lane_index,
                max_lane=self.config["lanes_count"],
                ego_vehicle=self.vehicle,
                env=self,
                action_config=self.config["pedestrians"]["action"],
                crossing_config=self.config["pedestrians"]["crossing"]
                )
            self.road.pedestrians.append(pedestrian)
            self.road.vehicles.append(pedestrian)
        self.pedestrians = self.road.pedestrians

    def _reward(self, action: Action) -> float:
        if self.vehicle.crashed:
            reward = self.config["collision_reward"] * 1
        elif not (self.vehicle.on_road or self.vehicle.position[0] >= self.config["road_length"]):
            reward = self.config["off_road_reward"] * 1
        elif self.vehicle.lane_index[2] in (0, self.config["lanes_count"] - 1):
            reward = self.config["off_lane_reward"] * 1
        else:
            rewards = self._rewards(action)
            reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
            if self.config["normalize_reward"]:
                low = sum((
                    self.config["low_speed_reward"],
                ))
                high = sum((
                    self.config["on_lane_reward"],
                    self.config["high_speed_reward"],
                    self.config["heading_reward"],
                ))
                reward = utils.lmap(reward, [low, high], self.config["reward_range"])
        if self.config["clip_reward"]:
            reward = np.clip(reward, self.config["reward_range"][0], self.config["reward_range"][1])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        rewards = {
            "on_lane_reward": 0,
            "high_speed_reward": 0,
            "low_speed_reward": 0,
            "heading_reward": 0,
            "prolong_static_reward": 0,
        }

        if not self.vehicle.crashed:
            # On-lane reward
            if self.vehicle_lane == self.vehicle.lane_index[2]:
                rewards["on_lane_reward"] = 1
            elif self.vehicle.lane_index[2] not in (0, self.config["lanes_count"]-1):
                rewards["on_lane_reward"] = 0.5
            self.vehicle_lane = self.vehicle.lane_index[2]

            # Heading reward
            heading = abs(self.vehicle.heading)
            if heading < np.pi / 2:
                rewards["heading_reward"] = 1 - utils.lmap(heading, [0, np.pi / 2], [0, 1])

            # Speed reward
            # Use forward speed, see https://github.com/Farama-Foundation/HighwayEnv/issues/268
            forward_speed = self.vehicle.velocity[0] #self.vehicle.speed * np.cos(self.vehicle.heading)
            forward_speed = max(forward_speed, 0.) # Fix precision loss
            if int(forward_speed) == 0:
                self.static_counter += 0
                if self.static_counter >= self.config["prolong_static_count"][0]:
                    if self.static_counter >= self.config["prolong_static_count"][1]:
                        rewards["prolong_static_reward"] = 1
                    else:
                        rewards["prolong_static_reward"] = utils.lmap(
                            self.static_counter,
                            self.config["prolong_static_count"],
                            [0, 1])
            else:
                self.static_counter = 0
                # Low speed reward
                low_speed = self.config["low_speed_range"]
                if forward_speed <= low_speed[1] and low_speed[1] > 0:
                    rewards["low_speed_reward"] = (low_speed[1] - forward_speed) / low_speed[1]
                # High speed reward
                low_speed, desired_speed, high_speed = self.config["high_speed_range"]
                if low_speed <= forward_speed <= desired_speed:
                    speed_tolerance = desired_speed - low_speed
                elif desired_speed <= forward_speed <= high_speed:
                    speed_tolerance = high_speed - desired_speed
                if low_speed <= forward_speed <= high_speed:
                    speed_diff = abs(forward_speed - desired_speed)
                    rewards["high_speed_reward"] = speed_diff / speed_tolerance
        return rewards

    def _is_terminated(self) -> bool:
        return (self.vehicle.crashed or
            self.config["offroad_terminal"] and not self.vehicle.on_road)
    
    def _is_truncated(self) -> bool:
        if self.config['duration']:
            return self.time >= self.config['duration']
        return False
    
    def get_performance(self):
        # Log last ego vehicle state
        d = {
            "collided": self.vehicle.crashed,
            "complete": not self.vehicle.on_road and self.vehicle.position[0] >= self.config["road_length"],
            "off_road": not self.vehicle.on_road,
            "position": self.vehicle.position,
            "lane_id": self.vehicle.lane_index[2],
            "velocity": self.vehicle.velocity,
            "heading": self.vehicle.heading,
        }
        # Count pedestrian crossed the road
        p_crossed = sum(map(lambda x: x.crossed, self.road.pedestrians))
        p_percent = p_crossed / self.config["pedestrians"]["count"] * 100.\
            if self.config["pedestrians"]["count"] > 0 else 0
        d.update({
            "pedestrians_crossed": (p_crossed, p_percent),
        })
        return d