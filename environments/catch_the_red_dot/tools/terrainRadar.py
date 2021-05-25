from typing import List, Any
from tools import Sensor

class TerrainRadar(Sensor):
    def __init__(self, terrain: List[Any]):
        self.terrain = terrain

    def activate(self) -> List[Any]:
        result = []
        for t in self.terrain:
            result.append(t['x'])
            result.append(t['y'])
            result.append(t['size'])
            result.append(t['type'])
        return result
