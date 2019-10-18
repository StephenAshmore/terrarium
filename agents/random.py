from .agent import Agent

class RandomAgent(Agent):
    def step(self) -> None:
        # take a random action, ignore all sensors
        print('ste')