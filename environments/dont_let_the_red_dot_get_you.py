from .environment import Environment
import numpy as np

class DontLetTheRedDotGetYou(Environment):
	def __init__(self) -> None:
		pass

	def step(self, params: np.ndarray) -> np.ndarray:
		print("step")

	def render(self) -> None:
		print("render")
