import random
import numpy as np
from .agent import Agent

class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.type = 'RandomAgent'

    def step(self) -> None:
        # Activate Sensors:
        nums: np.ndarray = []
        for i in range(0, len(self.sensors)):
            nums = nums + self.activate_sensor(i)
        # debug
        obs = np.array(nums)
        # take a random action, ignore all sensors
        if len(self.actuators) > 0:
            actuator_choice = random.randrange(0, len(self.actuators))
            action = np.empty([0])
            for i in range(0, 8):
                action = np.append(action, np.random.normal())
            self.activate_actuator(actuator_choice, action)
