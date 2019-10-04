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
        self.grid_shape = (15, 15)
        self.grid = None
        self.max_steps = 1000
        self.topleft, self.bottomright = np.array([0, 0]), np.array(list(self.grid_shape))

        self.head_position = None

        self.initial_length = 3
        self.directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        self.direction_pointer = 0

        self.snake_heading = None
        self.snake_body = None

        # render
        self.screen = None


    def step(self, action):
        action = 1 if action>1 else -1 if action<-1 else action
        # action = max(min(action, 1), -1)
        self.direction_pointer = (self.direction_pointer+action)%len(self.directions)
        self.snake_heading = self.directions[self.direction_pointer]
        head_y, head_x = new_head_yx = self.head_position + self.snake_heading

        if head_y<self.topleft[0] or head_y>=self.bottomright[0]\
                or head_x<self.topleft[1] or head_x>=self.bottomright[1]\
                or self.grid[head_y][head_x] == 1.\
                or self.current_step>=self.max_steps:
            return self.grid.flatten(), -1, True, {}

        if self.grid[head_y][head_x] == 2.:
            reward = 1
            self.grid[head_y][head_x] = 1.
            self.generate_new_food()
        else:
            tail_y, tail_x = self.snake_body.pop()
            self.grid[tail_y][tail_x] = 0.
            reward = 0

        self.grid[head_y][head_x] = 1.
        self.head_position = new_head_yx
        self.snake_body.appendleft(new_head_yx)

        self.current_step += 1
        return self.grid.flatten(), reward, False, {}

    def reset(self):
        self.grid = np.zeros(self.grid_shape)
        self.current_step = 0
        self.head_position = np.array([5, 10])
        self.direction_pointer = 0

        self.snake_heading = self.directions[self.direction_pointer]
        self.snake_body = collections.deque([self.head_position])
        self.grid[self.head_position[0]][self.head_position[1]] = 1.
        for i in range(1, self.initial_length):
            y, x = yx = self.head_position + i * self.snake_heading
            self.snake_body.append(yx)
            self.grid[y][x] = 1.

        self.direction_pointer = 2
        self.generate_new_food()
        return self.grid.flatten()

    def render(self, mode='human', close=False):
        if self.screen is None:
            pygame.init()
            self.size = self.width, self.height = 600, 600
            self.rect_width, self.rect_height = tuple(self.size / np.array(list(self.grid_shape)))
            self.screen = pygame.display.set_mode(self.size)

        pygame.draw.rect(self.screen, (0, 128, 255), pygame.Rect(0, 0, self.width, self.height))

        xs, ys = np.where(self.grid != 0.)
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
        # time.sleep(.333)

    def yx_coords(self, index):
        y, x = index // self.grid_shape[0], index % self.grid_shape[0]
        return y, x

    def generate_new_food(self):
        probs = 1-self.grid.flatten()
        probs /= probs.sum()
        ind = np.random.choice(len(probs), 1, p=probs)[0]
        y, x = ind//self.grid_shape[0], ind%self.grid_shape[0]
        self.grid[y][x] = 2.


