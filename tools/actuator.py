from abc import abstractmethod
import numpy as np

from .tool import Tool

class Actuator(Tool):
	ACTION_DIMS = 3

	def __init__(self) -> None:
		pass

	@abstractmethod
	def activate(self, params: np.ndarray) -> None:
		raise NotImplementedError('stub')
