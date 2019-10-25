from typing import List, Any
from abc import abstractmethod
import numpy as np

from tools import Actuator, Sensor

class Agent(object):
	def __init__(self) -> None:
		self.sensors: List[Sensor] = []
		self.actuators: List[Actuator] = []
		self.id: int = -1
		self.name: str = ''
		self.type: str = 'BaseAgent'

	def add_id(self, id: int) -> None:
		self.id = id
		for s in self.sensors:
			s.add_owner(self.id)
		for a in self.actuators:
			a.add_owner(self.id)

	def add_sensor(self, sensor: Sensor) -> int:
		sensor.add_owner(self.id)
		self.sensors.append(sensor)
		return len(self.sensors) - 1

	def drop_sensor(self, sensor_index: int) -> None:
		if sensor_index < len(self.sensors) and sensor_index >= -len(self.sensors):
			del self.sensors[sensor_index]

	def activate_sensor(self, sensor_index: int) -> np.ndarray:
		if sensor_index < len(self.sensors) and sensor_index >= -len(self.sensors):
			return self.sensors[sensor_index].activate()

	def add_actuator(self, actuator: Actuator) -> int:
		actuator.add_owner(self.id)
		self.actuators.append(actuator)
		return len(self.actuators) - 1

	def drop_actuator(self, actuator_index: int) -> None:
		if actuator_index < len(self.actuators) and actuator_index >= -len(self.actuators):
			del self.actuators[actuator_index]

	def activate_actuator(self, actuator_index: int, params: np.ndarray) -> None:
		if actuator_index < len(self.actuators) and actuator_index >= -len(self.actuators):
			return self.actuators[actuator_index].activate(params)

	def set_name(self, name: str) -> None:
		self.name = name

	def get_name(self) -> str:
		return self.name

	@abstractmethod
	def step(self) -> None:
		raise NotImplementedError('stub')
