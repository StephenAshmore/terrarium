from abc import abstractmethod
import numpy as np

class Tool(object):
	def __init__(self) -> None:
		pass

	@abstractmethod
	def activate(self, params: np.ndarray) -> np.ndarray:
		raise NotImplementedError('stub')
