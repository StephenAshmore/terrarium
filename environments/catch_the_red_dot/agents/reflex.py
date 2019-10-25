# Handcoded solution to catching the red dot
# Using the Radar tool, we always know where the red dot is.
# Assuming there is no terrain, a straight line is the fastest between two points.
# Follow the line.
import numpy as np

from agents import Agent

class ReflexAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.type = 'ReflexAgent'

    def step(self) -> None:
        # Get current location:
        loc = self.activate_sensor(0)
        # Get red dot location from radar:
        red = self.activate_sensor(1)

        x_move = 0
        if red[0] > loc[0]:
            x_move = 1
        elif red[0] < loc[0]:
            x_move = -1

        y_move = 0
        if red[1] > loc[1]:
            y_move = 1
        elif red[1] < loc[1]:
            y_move = -1

        actions = np.zeros([8])
        actions[0] = x_move
        actions[1] = y_move
        self.activate_actuator(0, actions)