from typing import Tuple, List, Any
import pygame
from environments import Environment
import numpy as np
import pygame.locals as pg
import random
from agents import Agent

class CatchTheRedDot(Environment):
	def __init__(self) -> None:
        # View stuff
		screen_size = (800,600)
		self.screen = pygame.display.set_mode(screen_size, 32)
		self.red = pygame.image.load("environments/resources/red.png")
		self.blue = pygame.image.load("environments/resources/blue.png")

        # Model stuff
		self.rect = pygame.Rect(0, 0, 64, 64)
		self.dest_x = 0
		self.dest_y = 0
		self.agents: List[Agent] = []
		self.positions: List[Any] = []
		self.max_agents = 3

	def add_agent(self, a: Agent) -> None:
		if len(self.agents) < self.max_agents:
			self.agents.append(a)
			self.positions.append({ 'x': random.randrange(0, 64), 'y': random.randrange(0,64) })
		else:
			raise ValueError(f'You cannot add more than {self.max_agents} agents!')

	def getLocation(self, agent_id: int) -> np.ndarray:
		return self.positions[agent_id]

    # Advance the model
	def step(self, params: np.ndarray) -> np.ndarray:
		if self.rect.left < self.dest_x:
			self.rect.left += 5
		if self.rect.left > self.dest_x:
			self.rect.left -= 5
		if self.rect.top < self.dest_y:
			self.rect.top += 5
		if self.rect.top > self.dest_y:
			self.rect.top -= 5

	def render(self) -> None:
        # Draw the screen
		self.screen.fill([0,200,100])
		self.screen.blit(self.red, self.rect)
		for p in self.positions:
			self.screen.blit(self.blue, pygame.Rect(p['x'], p['y'], p['x'] + 64, p['y'] + 64))
		pygame.display.flip()
