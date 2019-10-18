from typing import Tuple, List
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.locals as pg
import time
from abc import abstractmethod
import numpy as np

class Tool(object):
	def __init__(self) -> None:
		pass

	@abstractmethod
	def activate(self, params: np.ndarray) -> np.ndarray:
		raise NotImplementedError('stub')

class Agent(object):
	def __init__(self) -> None:
		self.tools: List[Tool] = []

	def add_tool(self, tool: Tool) -> None:
		self.tools.append(tool)

	def drop_tool(self, tool_index: int) -> None:
		if tool_index < len(self.tools) and tool_index >= -len(self.tools):
			self.tools.remove(tool_index)

	def activate_tool(self, tool_index: int, params: np.ndarray) -> None:
		if tool_index < len(self.tools) and tool_index >= -len(self.tools):
			self.tools[tool_index].activate(params)

class Model(object):
	def __init__(self) -> None:
		self.rect = pygame.Rect(0, 0, 80, 59)
		self.dest_x = 0
		self.dest_y = 0

	def update(self) -> None:
		if self.rect.left < self.dest_x:
			self.rect.left += 1
		if self.rect.left > self.dest_x:
			self.rect.left -= 1
		if self.rect.top < self.dest_y:
			self.rect.top += 1
		if self.rect.top > self.dest_y:
			self.rect.top -= 1

	def set_dest(self, pos: Tuple[int, int]) -> None:
		self.dest_x = pos[0]
		self.dest_y = pos[1]

class View(object):
	def __init__(self, model: Model) -> None:
		screen_size = (800,600)
		self.screen = pygame.display.set_mode(screen_size, 32)
		self.turtle_image = pygame.image.load("turtle.png")
		self.model = model
		self.model.rect = self.turtle_image.get_rect()

	def update(self) -> None:
		self.screen.fill([0,200,100])
		self.screen.blit(self.turtle_image, self.model.rect)
		pygame.display.flip()

class Controller(object):
	def __init__(self, model: Model) -> None:
		self.model = model
		self.keep_going = True

	def update(self) -> None:
		for event in pygame.event.get():
			if event.type == pg.QUIT:
				self.keep_going = False
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_ESCAPE:
					self.keep_going = False
			elif event.type == pygame.MOUSEBUTTONUP:
				self.model.set_dest(pygame.mouse.get_pos())
		keys = pygame.key.get_pressed()
		if keys[pg.K_LEFT]:
			self.model.dest_x -= 1
		if keys[pg.K_RIGHT]:
			self.model.dest_x += 1
		if keys[pg.K_UP]:
			self.model.dest_y -= 1
		if keys[pg.K_DOWN]:
			self.model.dest_y += 1

print("Use the arrow keys to move. Press Esc to quit.")
pygame.init()
m = Model()
v = View(m)
c = Controller(m)
while c.keep_going:
	c.update()
	m.update()
	v.update()
	time.sleep(0.04)
print("Goodbye")
