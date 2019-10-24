from typing import List, Any
import numpy as np

from tools import Actuator

class Thruster(Actuator):
    def __init__(self, positions: List[Any]):
        self.positions: List[Any] = positions

    def activate(self, params: np.ndarray) -> None:
        x_move = min(max(params[0], -1), 1)
        y_move = min(max(params[1], -1), 1)
        self.positions[self.owner] = [
            self.positions[self.owner][0] + x_move,
            self.positions[self.owner][1] + y_move,
        ]