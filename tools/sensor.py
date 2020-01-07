from abc import abstractmethod
import numpy as np

from .tool import Tool

class Sensor(Tool):
	SENSOR_DIMS = 3

	def __init__(self) -> None:
		pass

	@abstractmethod
	def activate(self) -> np.ndarray:
		raise NotImplementedError('stub')
