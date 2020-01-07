from typing import List
import random
import numpy as np
from .agent import Agent
from tools import Actuator

class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.type = 'RandomAgent'

    def step(self) -> None:
        # Activate Sensors:
        nums: List[np.ndarray] = []
        for i in range(1, len(self.sensors)): # skip the first sensor, which is the thermometer
            nums.append(self.activate_sensor(i))
        obs = np.array(nums)

        # Take random actions, ignoring all sensors
        for i in range(len(self.actuators)):
            action = np.random.normal(0., 1., (Actuator.ACTION_DIMS,))
            self.activate_actuator(i, action)
