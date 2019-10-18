from typing import Tuple
from abc import abstractmethod
import numpy as np

class Environment(object):
	def __init__(self) -> None:
		pass

	@abstractmethod
	def step(self, params: np.ndarray) -> np.ndarray:
		raise NotImplementedError('stub')

	@abstractmethod
	def render(self) -> None:
		raise NotImplementedError('stub')

	def on_click(self, pos: Tuple[int, int]) -> None:
		pass
