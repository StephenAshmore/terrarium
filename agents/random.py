import random

from .agent import Agent

class RandomAgent(Agent):
    def step(self) -> None:
        # take a random action, ignore all sensors
        tool_choice = random.randrange(0, len(self.tools))
        self.activate_tool(tool_choice, [0, 0, 0])
        print('step')