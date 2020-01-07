from typing import List, Any
import numpy as np
from tools import Sensor
import numpy as np

class Locator(Sensor):
    def __init__(self, positions: List[Any]):
        self.positions: List[List[Any]] = positions

    # Returns the position of the calling agent
    def activate(self) -> np.ndarray:
        obs = np.zeros(Sensor.SENSOR_DIMS)
        obs[0:2] = self.positions[self.owner]
        return obs
