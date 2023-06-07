import random
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
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed, predition_type)
        self.check_collisions = False

    def act(self, action: dict | str = None) -> None:
        # TO-DO:
        return super().act({
            "acceleration": random.random(),
            "steering": random.random(),
        })