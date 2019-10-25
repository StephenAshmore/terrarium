from typing import Tuple, List
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.locals as pg
import time
from abc import abstractmethod
from environments import Environment, CatchTheRedDot
from agents import RandomAgent

from environments.catch_the_red_dot.agents import ReflexAgent

class Controller(object):
	def __init__(self, env: Environment) -> None:
		self.env = env
		self.keep_going = True

	def update(self) -> None:
		for event in pygame.event.get():
			if event.type == pg.QUIT:
				self.keep_going = False
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_ESCAPE:
					self.keep_going = False
			elif event.type == pygame.MOUSEBUTTONUP:
				self.env.on_click(pygame.mouse.get_pos())
		env.step()
		env.render()

if __name__ == '__main__':
	print("Use the arrow keys to move. Press Esc to quit.")
	pygame.init()

	env = CatchTheRedDot()
	env.add_agent(RandomAgent())
	env.add_agent(ReflexAgent())

	c = Controller(env)
	while c.keep_going:
		c.update()
		time.sleep(0.04)
	print("Goodbye")
