import tools.tool

class Location2d(Tool):
    def __init__(self) -> None:
        pass

    def activate(self, params: np.ndarray) -> np.ndarray:
        return [1, 2]