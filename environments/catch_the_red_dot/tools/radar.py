from typing import List, Any
from tools import Sensor

class Radar(Sensor):
    def __init__(self, destination: List[Any]):
        self.destination = destination

    def activate(self) -> List[Any]:
        return self.destination