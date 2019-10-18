from typing import Tuple
import pygame
from environments import Environment
import numpy as np
import pygame.locals as pg

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
        # Handle key presses
		keys = pygame.key.get_pressed()
		if keys[pg.K_LEFT]:
			self.dest_x -= 5
		if keys[pg.K_RIGHT]:
			self.dest_x += 5
		if keys[pg.K_UP]:
			self.dest_y -= 5
		if keys[pg.K_DOWN]:
			self.dest_y += 5

        # Draw the screen
		self.screen.fill([0,200,100])
		self.screen.blit(self.red, self.rect)
		pygame.display.flip()

    # Handle mouse clicks
	def on_click(self, pos: Tuple[int, int]) -> None:
		self.dest_x = pos[0]
		self.dest_y = pos[1]
