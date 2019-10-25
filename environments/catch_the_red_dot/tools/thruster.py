from typing import List, Any
import numpy as np

from tools import Actuator

class Thruster(Actuator):
    def __init__(self, positions: List[Any], terrain_map: np.ndarray):
        self.positions: List[Any] = positions
        self.terrain_map: np.ndarray = terrain_map
        self.movement_modifier = 3

    def activate(self, params: np.ndarray) -> None:
        x_move = min(max(params[0], -1), 1)
        y_move = min(max(params[1], -1), 1)

        new_x = int(self.positions[self.owner][0] + x_move)
        new_y = int(self.positions[self.owner][1] + y_move)
        print(new_x, new_y, self.terrain_map.shape[1], self.terrain_map.shape[0])
        if new_x >= 0 and new_x < self.terrain_map.shape[1] and new_y >= 0 and new_y < self.terrain_map.shape[0]:
            print(f'Moving at speed {self.terrain_map[new_y, new_x]}')
            self.positions[self.owner] = [
                self.positions[self.owner][0] + (x_move * self.terrain_map[new_y, new_x] * self.movement_modifier),
                self.positions[self.owner][1] + (y_move * self.terrain_map[new_y, new_x] * self.movement_modifier),
            ]
