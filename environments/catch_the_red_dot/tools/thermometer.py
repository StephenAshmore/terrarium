from typing import List, Any
import numpy as np
from tools import Sensor
import numpy as np
import math

class Thermometer(Sensor):
    def __init__(self, positions: List[Any], destination: List[Any]):
        self.positions: List[List[Any]] = positions
        self.destination = destination

    # Returns a value indicating how close the agent is to the destination
    def activate(self) -> np.ndarray:
        pos = self.positions[self.owner]
        dest = self.destination
        obs = np.zeros(Sensor.SENSOR_DIMS)
        obs[0] = 1.0 / math.sqrt((pos[0] - dest[0]) ** 2 + (pos[1] - dest[1]) ** 2)
        return obs
