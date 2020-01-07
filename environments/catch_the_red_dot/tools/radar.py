from typing import List, Any
from tools import Sensor
import numpy as np

class Radar(Sensor):
    def __init__(self, destination: List[Any]):
        self.destination = destination

    # Returns the position of the target being chased
    def activate(self) -> np.ndarray:
        obs = np.zeros(Sensor.SENSOR_DIMS)
        obs[0:2] = self.destination
        return obs
