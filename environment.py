class Environment(object):
	def __init__(self) -> None:
		pass

	@abstractmethod
	def step(self, params: np.ndarray) -> np.ndarray:
		raise NotImplementedError('stub')

	@abstractmethod
	def render(self) -> None:
		raise NotImplementedError('stub')
