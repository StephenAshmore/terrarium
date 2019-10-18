class Agent(object):
	def __init__(self) -> None:
		self.tools: List[Tool] = []

	def add_tool(self, tool: Tool) -> None:
		self.tools.append(tool)

	def drop_tool(self, tool_index: int) -> None:
		if tool_index < len(self.tools) and tool_index >= -len(self.tools):
			self.tools.remove(tool_index)

	def activate_tool(self, tool_index: int, params: np.ndarray) -> None:
		if tool_index < len(self.tools) and tool_index >= -len(self.tools):
			self.tools[tool_index].activate(params)
