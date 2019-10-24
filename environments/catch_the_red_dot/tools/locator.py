from typing import List, Any
import numpy as np
from tools import Sensor

class Locator(Sensor):
    def __init__(self, positions: List[Any]):
        self.positions: List[Any] = positions

    def activate(self) -> List[Any]:
        return self.positions[self.owner]
