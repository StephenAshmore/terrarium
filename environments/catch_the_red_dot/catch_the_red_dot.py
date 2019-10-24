from typing import Tuple, List, Any
import pygame
from environments import Environment
import numpy as np
import pygame.locals as pg
import random
import time

from agents import Agent
from .tools import Locator, Radar, Thruster


class CatchTheRedDot(Environment):
    def __init__(self) -> None:
        # View stuff
        self.screen_size = (800, 600)
        self.screen = pygame.display.set_mode(self.screen_size, 32)
        self.red = pygame.image.load("environments/resources/red.png")
        self.blue = pygame.image.load("environments/resources/blue.png")

        # Model stuff
        self.destination = [random.randrange(0, self.screen_size[0] - 64), random.randrange(0, self.screen_size[1] - 64)]
        self.rect = pygame.Rect(
            self.destination[0],
            self.destination[1],
            self.destination[0] + 64,
            self.destination[1] + 64
        )
        self.agents: List[Agent] = []
        self.positions: List[Any] = []
        self.max_agents = 3
        self.current_agents = 0

    def reset(self) -> None:
        # Model stuff
        self.destination[0] = random.randrange(0, self.screen_size[0] - 64)
        self.destination[1] = random.randrange(0, self.screen_size[1] - 64)
        self.rect = pygame.Rect(
            self.destination[0],
            self.destination[1],
            self.destination[0] + 64,
            self.destination[1] + 64
        )
        self.max_agents = 3
        for i in range(0, len(self.positions)):
            self.positions[i] = [random.randrange(
                0, self.screen_size[0] - 64), random.randrange(0, self.screen_size[1] - 64)]

    def add_agent(self, a: Agent) -> None:
        if len(self.agents) < self.max_agents:
            self.positions.append([random.randrange(
                0, self.screen_size[0] - 64), random.randrange(0, self.screen_size[1] - 64)])
            a.add_sensor(Locator(self.positions))
            a.add_sensor(Radar(self.destination))
            a.add_actuator(Thruster(self.positions))
            a.add_id(self.current_agents)
            self.agents.append(a)
            self.current_agents = self.current_agents + 1
        else:
            raise ValueError(
                f'You cannot add more than {self.max_agents} agents!')

    # Advance the model
    def step(self) -> None:
        for a in self.agents:
            a.step()
        winner = self.check_win()
        if winner != -1:
            print(f'Winner winner chicken dinner for {winner}')
            time.sleep(3)
            self.reset()

    def render(self) -> None:
        # Draw the screen
        self.screen.fill([0, 200, 100])
        self.screen.blit(self.red, self.rect)
        for p in self.positions:
            self.screen.blit(self.blue, pygame.Rect(
                p[0], p[1], p[0] + 64, p[1] + 64))
        pygame.display.flip()

    def check_win(self) -> int:
        agent_id = 0
        for p in self.positions:
            if p[0] > self.destination[0] - 1 and p[0] < self.destination[0] + 1:
                if p[1] > self.destination[1] - 1 and p[1] < self.destination[1] + 1:
                    return agent_id
            agent_id = agent_id + 1
        return -1
