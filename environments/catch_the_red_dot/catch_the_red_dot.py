from typing import Tuple, List, Any, Mapping
import pygame
from environments import Environment
import numpy as np
import pygame.locals as pg
import random
import time
import math

from agents import Agent
from .tools import Locator, Radar, Thermometer, TerrainRadar, Thruster


class CatchTheRedDot(Environment):
    def __init__(self) -> None:
        # View stuff
        self.screen_size = (800, 600)
        self.screen = pygame.display.set_mode(self.screen_size, 32)
        self.red = pygame.image.load("environments/resources/red.png")
        self.blue = pygame.image.load("environments/resources/blue.png")
        self.font = pygame.font.SysFont(None, 12)

        # Model stuff
        self.terrain_areas: List[Mapping[str, Any]] = []
        self.terrain_types = ['normal', 'mountain', 'forest']
        self.terrain_colors = [[0, 200, 100], [190, 190, 190], [0, 255, 0]]
        self.terrain_speed = [1, 0.1, 0.5]
        self.terrain_map = np.ones([self.screen_size[1], self.screen_size[0]])
        self.destination = [0, 0]
        self.create_terrain()
        self.place_red_dot()
        self.rect = pygame.Rect(
            self.destination[0],
            self.destination[1],
            self.destination[0] + 64,
            self.destination[1] + 64
        )
        self.agents: List[Agent] = []
        self.render_names: List[Any] = []
        self.positions: List[List[float]] = []
        self.max_agents = 3
        self.current_agents = 0

    def reset(self) -> None:
        # Model stuff
        self.clear_terrain()
        self.create_terrain()
        self.place_red_dot()
        self.rect = pygame.Rect(
            self.destination[0],
            self.destination[1],
            self.destination[0] + 64,
            self.destination[1] + 64
        )
        self.max_agents = 3
        for i in range(0, len(self.positions)):
            self.positions[i] = [random.randrange(
                0, self.screen_size[0] - 64), random.randrange(0, self.screen_size[1] - 64)]
        for agent in self.agents:
            agent.reset()

    def add_agent(self, a: Agent) -> None:
        if len(self.agents) < self.max_agents:
            self.positions.append([random.randrange(
                0, self.screen_size[0] - 64), random.randrange(0, self.screen_size[1] - 64)])
            a.add_sensor(Thermometer(self.positions, self.destination))
            a.add_sensor(Locator(self.positions))
            a.add_sensor(Radar(self.destination))
            a.add_actuator(Thruster(self.positions, self.terrain_map))
            a.add_id(self.current_agents)
            self.agents.append(a)
            self.render_names.append(self.font.render(
                a.get_name() + a.type, True, (255, 255, 255)))
            self.current_agents = self.current_agents + 1
        else:
            raise ValueError(
                f'You cannot add more than {self.max_agents} agents!')

    # Advance the model
    def step(self) -> None:
        for a in self.agents:
            a.step()
        winner = self.check_win()
        if winner != -1:
            print(f'Winner winner chicken dinner for {winner}')
            self.reset()

    def render(self) -> None:
        # Draw the screen
        # Default terrain fill:
        self.screen.fill(self.terrain_colors[0])
        # Draw terrain:
        for t in self.terrain_areas:
            color = self.terrain_colors[t['type']]
            pygame.draw.circle(self.screen, color, [t['x'], t['y']], t['size'])

        self.screen.blit(self.red, self.rect)
        i = 0
        for p in self.positions:
            self.screen.blit(self.blue, pygame.Rect(
                p[0], p[1], p[0] + 64, p[1] + 64))
            self.screen.blit(self.render_names[i], (p[0] + 4, p[1] + 30))
            i = i + 1
        pygame.display.flip()

    def check_win(self) -> int:
        agent_id = 0
        for p in self.positions:
            x1 = p[0]
            x2 = self.destination[0]
            y1 = p[1]
            y2 = self.destination[1]
            if self.circle_collision(x1, x2, y1, y2, 30, 30):
                return agent_id
            agent_id = agent_id + 1
        return -1

    def create_terrain(self) -> None:
        terrain_pools = random.randrange(2, 5)
        for i in range(0, terrain_pools):
            t = self.create_terrain_area()
            type = random.randrange(1, len(self.terrain_types))
            self.terrain_areas.append({
                'x': t[0],
                'y': t[1],
                'size': t[2],
                'type': type
            })
            self.update_terrain_map(t[0], t[1], t[2], type)

    def update_terrain_map(self, x: int, y: int, size: int, type: int) -> None:
        for c_y in range(-size, size):
            for c_x in range(-size, size):
                if abs(c_y) + abs(c_x) <= size:
                    f_x = x + c_x
                    f_y = y + c_y
                    if f_y >= 0 and f_y < self.screen_size[1] and f_x >= 0 and f_x < self.screen_size[0]:
                        self.terrain_map[f_y][f_x] = self.terrain_speed[type]

    def create_terrain_area(self) -> Tuple[int, int, int]:
        t_x = random.randrange(0, self.screen_size[0])
        t_y = random.randrange(0, self.screen_size[1])

        t_size = random.randrange(60, 160)

        if len(self.terrain_areas) > 0:
            recompute = False
            for t in self.terrain_areas:
                if self.circle_collision(
                    t_x, t['x'],
                    t_y, t['y'],
                    t_size, t['size']
                ):
                    recompute = True
                    break
            if recompute:
                return self.create_terrain_area()

        return (t_x, t_y, t_size)

    def place_red_dot(self) -> None:
        self.destination[0] = random.randrange(0, self.screen_size[0] - 64)
        self.destination[1] = random.randrange(0, self.screen_size[1] - 64)
        replace = False
        for t in self.terrain_areas:
            if self.terrain_speed[t['type']] == 0:
                if self.circle_collision(
                    self.destination[0] + 32, t['x'],
                    self.destination[1] + 32, t['y'],
                    32, t['size']
                ):
                    replace = True
                    break
        if replace:
            self.place_red_dot()

    def circle_collision(self, x1: float, x2: float, y1: float, y2: float, size1: float, size2: float) -> bool:
        distance = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        return distance < size1 + size2

    def clear_terrain(self) -> None:
        self.terrain_areas.clear()
        for i in range(0, self.terrain_map.shape[0]):
            for j in range(0, self.terrain_map.shape[1]):
                self.terrain_map[i, j] = 1
