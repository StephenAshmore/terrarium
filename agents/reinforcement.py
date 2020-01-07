from typing import List
import numpy as np
from .agent import Agent
from tools import Actuator
from tools import Sensor
from .qlearner import qlearner

class ReinforcementAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.num_sensors = 3
        self.num_actuators = 1
        self.type = 'ReinforcementAgent'
        self.q = qlearner(self.num_sensors * (Sensor.SENSOR_DIMS - 1), self.num_actuators * Actuator.ACTION_DIMS)
        self.arrived_here_deliberately = False

    def reset(self) -> None:
        self.arrived_here_deliberately = False

    def step(self) -> None:
        # Check assumptions
        assert self.num_sensors == len(self.sensors)
        assert self.num_actuators == len(self.actuators)

        # Activate Sensors:
        nums: List[np.ndarray] = []
        for i in range(1, len(self.sensors)): # skip the first sensor, which is the thermometer
            nums.append(self.activate_sensor(i))
        obs = np.concatenate(nums)

        # Learn from experience
        if self.arrived_here_deliberately:
            reward = self.activate_sensor(0)[0]
            self.q.learn_from_reward(obs, reward)
        self.arrived_here_deliberately = True

        # Perform the next action
        actions = self.q.choose_next_action(obs)
        assert(actions.shape[0] == len(self.actuators) * Actuator.ACTION_DIMS)
        for i in range(len(self.actuators)):
            pos = Actuator.ACTION_DIMS * i
            self.activate_actuator(i, actions[pos : pos + Actuator.ACTION_DIMS])
