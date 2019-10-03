import numpy as np
import collections
import pygame
import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class SnakeEnv(gym.Env):
    metadata = {
            'render.modes': ['human']
            }

    def __init__(self):
        self.grid = np.zeros((15, 15), np.int)
        self.topleft, self.bottomright = np.array([0, 0]), np.array([15, 15])
        self.head_position = np.array([5, 10])
        self.initial_length = 3
        self.directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], np.int)
        self.direction_pointer = 0
        self.action = 0
        self.snake_heading = self.directions[self.direction_pointer]
        self.snake_body = collections.deque([self.head_position])
        for i in range(self.initial_length-1):
            self.snake_body.append(self.head_position+i*self.snake_heading)

        self.direction_pointer = 2

        for i in range(len(self.snake_body)):
            y, x = yx = self.snake_body.popleft()
            self.grid[y][x] = 1
            self.snake_body.append(yx)

        # render
        pygame.init()
        self.size = self.width, self.height = 600, 600
        self.rect_width, self.rect_height = tuple(self.size/np.array(list(self.grid.shape)))
        self.screen = None

    def step(self, action):
        self.action = np.clip(action, -1, 1)
        self.direction_pointer = (self.direction_pointer+self.action)%len(self.directions)
        self.snake_heading = self.directions[self.direction_pointer]
        head_y, head_x = new_head_yx = self.head_position + self.snake_heading
        if np.any(new_head_yx<self.topleft)\
                or np.any(new_head_yx>=self.bottomright)\
                or self.grid[head_y][head_x] == 1:
            return self.grid, -1, True, {}

        if self.grid[head_y][head_x] == 2:
            reward = 1
        else:
            tail_y, tail_x = self.snake_body.pop()
            self.grid[tail_y][tail_x] = 0
            reward = 0

        self.grid[head_y][head_x] = 1
        self.head_position = new_head_yx
        self.snake_body.appendleft(new_head_yx)

        return self.grid, reward, False, {}

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.size)

        pygame.draw.rect(self.screen, (0, 128, 255), pygame.Rect(0, 0, self.width, self.height))

        xs, ys = np.where(self.grid != 0)
        for y, x in zip(xs, ys):
            v = self.grid[y][x]
            if v == 1:
                if (x+y)%2==0:
                    c = (128, 128, 0)
                else:
                    c = (128, 255, 0)
            else:
                c = (255, 128, 0)
            pygame.draw.rect(self.screen, c,
                             pygame.Rect(x*self.rect_width, y*self.rect_height, self.rect_width, self.rect_height))

        y, x = self.head_position
        pygame.draw.rect(self.screen, (255, 0, 0),
                         pygame.Rect(x * self.rect_width, y * self.rect_height, self.rect_width, self.rect_height))

        pygame.display.flip()
        time.sleep(.33)

