from abc import abstractmethod
import numpy as np

from .tool import Tool

class Actuator(Tool):
	def __init__(self) -> None:
		pass

	@abstractmethod
	def activate(self, params: np.ndarray) -> None:
		raise NotImplementedError('stub')
