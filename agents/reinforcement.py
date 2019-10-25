import numpy as np
from .agent import Agent

class ReinforcementAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.type = 'ReinforcementAgent'
    
    def step(self) -> None:
        # Get reward
        print('Step Reinforcement Agent')
