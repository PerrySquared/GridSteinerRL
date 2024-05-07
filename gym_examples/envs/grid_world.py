from gymnasium import spaces
from .utils.dataset_extractor import get_coords_dataset
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
import numpy as np
import torch
import torchvision.transforms as T
import random
import copy
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

TERMINAL_CELL = 2
PATH_CELL = 1 

RENDER_EACH = 100000
RESET_EACH = 1000

random.seed(11)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 0.5}

    def __init__(self, render_mode=None, size=32):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        self._target_locations_copy = []
        self._agent_location_copy = []
        self._target_locations = []
        self._agent_location = []
        self.env_steps = 0
        
        # self.observation_space = spaces.Box(0, 1, shape=(96, 96), dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                # "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                # "target_locations": spaces.Box(0, 5, shape=(self.size, self.size), dtype=np.float64),
                "target_matrix": spaces.Box(0, 1, shape=(64, 64), dtype=np.float64),
                "target_list": spaces.Box(0, self.size - 1, shape=(5,2), dtype=np.int64),
                # "targets_relative_line": spaces.Discrete(5),
                # "targets_relative_general": spaces.Box(0, 1, shape=(4,), dtype=int),
                "targets_left": spaces.Discrete(6)
            }
        )
        
        self.action_space = spaces.MultiDiscrete(np.array([5, 5]))
        
        render_mode = "human"
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
        
        
    def reset(self, seed=None, options=None):
        self.matrix = np.zeros((self.size,self.size), dtype=np.int64)
        self.iterations = 0
        self.total_footprint = 0
        self.initial_targets_amount = 1
        self.stagnate_counter = 0
        self.previous_action = [-1, -1]
        
        
        if self.env_steps % RESET_EACH == 0:    # get new random env every 100 envs
            self._target_locations_copy = get_coords_dataset()
            
        self.env_steps += 1
        
        self._target_locations = copy.deepcopy(self._target_locations_copy)
        
        for _target_location in self._target_locations:
            self.matrix[_target_location[0], _target_location[1]] = TERMINAL_CELL

        observation = self._get_obs()
        info = self._get_info()
        
        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            if self.render_mode == "human":
                self._render_frame()
            if self.render_mode == "rgb_array":
                self._render_frame_as_rgb_array()


        return observation, info
           

    def step(self, action):
        # check if exited
        if self.env_steps % RENDER_EACH == 0:
            if self.render_mode == "human" or self.render_mode == "rgb_array":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

        self.iterations += 1
        
        reward = 0
        terminated = False     
        truncated = False
        
        self._move(action) # update the position 

        reward, terminated, truncated = self.game_over_check()

        same_elem = np.equal(action[0], action[1])
        # if both elements picked by action are the same to prevent just picking the terminals
        if same_elem.all(): 
            reward -= 1

        action_continuous = np.equal(np.sort(self.previous_action), np.sort(action))
        # if current action doesnt include one of the previous actions (to prevent not connected paths between two pairs of terminals) unless the first iteration
        if self.iterations > 1 and not action_continuous.any(): 
            reward -= 1
            
        # if connects same terminals in a different order
        if action_continuous.all():
            reward -= 1
        
        self.previous_action = copy.deepcopy(action)
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            print(reward)
            if self.render_mode == "human":
                self._render_frame()
            if self.render_mode == "rgb_array":
                self._render_frame_as_rgb_array()

        # 6. return game over and score
        return observation, reward, terminated, truncated, info

    
    def _move(self, action):
        first_terminal = self._target_locations[action[0]]
        second_terminal = self._target_locations[action[1]]     
        grid_path = self.create_manhattan_path(first_terminal, second_terminal)
        
        self.matrix = self.concatenate_with_override(self.matrix, grid_path)
        
    def game_over_check(self):
        reward = 0
        terminated = False
        truncated = False
        
        footprint = np.count_nonzero(self.matrix == PATH_CELL) # how many path cells are there
            
        # comparing footprint after current step with the previous one (might want to add && not_moved == False to consider intersections next to each other)
        if footprint > self.total_footprint: 
            reward -= (1 / (self.size * 2) ) * (footprint - self.total_footprint) # roughly 1/30th of the max footprint in a single step so max negative per step is 0.99 - can be calculated as 1/self.size if size is variable
            self.total_footprint = copy.deepcopy(footprint)
        else:
            reward -= 1
            
        if np.count_nonzero(self.matrix == TERMINAL_CELL) == 0: # successful game over
            terminated = True
            # reward = 1 # was uncommented
            
        if self.iterations > 10:
            reward = -1
            truncated = True

        return reward, terminated, truncated # score is less with each step
        
    def concatenate_with_override(self, arr1, arr2):
        result = []
        for i in range(len(arr1)):
            row = []
            for j in range(len(arr1[i])):
                if arr2[i][j] != 0:
                    row.append(arr2[i][j])
                else:
                    row.append(arr1[i][j])
            result.append(row)
            
        return np.asanyarray(result, dtype=int)
    
    def create_manhattan_path(self, start, end):
        # Create an empty grid initialized with zeros
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        
        # Extract coordinates for clarity
        y1, x1 = start
        y2, x2 = end
        
        # Mark the Manhattan path with 1s
        # Move horizontally to the correct column
        for x in range(min(x1, x2), max(x1, x2) + 1):
            grid[y1][x] = PATH_CELL
        # Move vertically to the correct row
        for y in range(min(y1, y2), max(y1, y2) + 1):
            grid[y][x2] = PATH_CELL
        
        return grid


    """observation and info methods"""

    def _get_obs(self):
        # print(self._agent_location,"\n", self.matrix,"==============")
        # print({"targets_relative_line": self.check_for_target_line(), "targets_relative_general": self.check_for_target_general()})
        output_shape = (64, 64)
        padding = ((3, 3), (3, 3))  # Padding of 2 rows and 2 columns on each side
        
        output_array = self.resize(np.divide(self.matrix, 2), output_shape, padding)
        # return output_array
        return {
            # "agent": self._agent_location,
            # "target_locations": self.get_matrix_with_targets(),
            "target_matrix": output_array,
            "target_list": np.array(self._target_locations, dtype=np.int64),
            # "targets_relative_line": self.check_for_target_line(),
            # "targets_relative_general": self.check_for_target_general(),
            "targets_left": np.count_nonzero(self.matrix == TERMINAL_CELL)
            }

    def resize(self, array, output_shape, padding):
        # Get dimensions of original and new array
        old_shape = array.shape
        rows_scale = output_shape[0] // old_shape[0]
        cols_scale = output_shape[1] // old_shape[1]
        
        # Repeat values in rows and columns
        repeated_rows = np.repeat(array, rows_scale, axis=0)
        repeated_cols = np.repeat(repeated_rows, cols_scale, axis=1)
        
        # Trim to desired shape
        resized_array = repeated_cols[:output_shape[0], :output_shape[1]]
        
        # padded_array = np.pad(resized_array, padding, mode='constant', constant_values=0)
        
        return resized_array
        
    
    def _get_info(self):
        return {
            # "agent": self._agent_location,
            "target_matrix": self.matrix
            }
    
    
    """extra observation space methods"""
    
    def get_matrix_with_targets(self):
        matrix = np.zeros((self.size, self.size))
        for target in self._target_locations:
            matrix[target[0], target[1]] = TERMINAL_CELL
            
        return matrix
    


    """render methods"""

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # draw targets
        for _target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * _target_location,
                    (pix_square_size, pix_square_size),
                ),
            )
                    
        # draw the global elements     
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                color = None
                if self.matrix[i, j] == PATH_CELL:
                    color = (0, 0, 150)
                if color:
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

                    
        font = pygame.font.Font(None, 24)  # Choose a font and size
        text_surface = font.render(str(self.env_steps), True, (0, 0, 255))  # Render the text
        canvas.blit(text_surface, (10, 10))  # Blit the text onto the canvas

        
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            if self.env_steps % 500 == 0: # on every N env step slow down rendering
                self.clock.tick(1)
            else:
                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )



    # def _render_frame_as_rgb_array(self): # not working yet
        
    #     if self.window is None and self.render_mode == "rgb_array":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode(
    #             (self.window_size, self.window_size)
    #         )
    #     if self.clock is None and self.render_mode == "rgb_array":
    #         self.clock = pygame.time.Clock()
            
    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels

    #     # Draw the target
    #     for _target_location in self._target_locations:
    #         pygame.draw.rect(
    #             canvas,
    #             (255, 0, 0),
    #             pygame.Rect(
    #                 pix_square_size * _target_location,
    #                 (pix_square_size, pix_square_size),
    #             ),
    #         )

    #     font = pygame.font.Font(None, 24)  # Choose a font and size
    #     text_surface = font.render(str(self.env_steps), True, (0, 0, 255))  # Render the text
    #     canvas.blit(text_surface, (10, 10))  # Blit the text onto the canvas

    #     # Draw the agent and other elements
    #     for i in range(len(self.matrix)):
    #         for j in range(len(self.matrix)):
    #             color = None
    #             if self.matrix[i, j] == PATH_CELL:
    #                 color = (0, 0, 150)
    #             if color:
    #                 pygame.draw.rect(
    #                     canvas,
    #                     color,
    #                     pygame.Rect(
    #                         pix_square_size * np.array([i, j]),
    #                         (pix_square_size, pix_square_size),
    #                     ),
    #                 )

    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         int(pix_square_size / 3),
    #     )

    #     # Add gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=1,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=1,
    #         )

    #     self.clock.tick(self.metadata["render_fps"])
        
    #     return np.transpose(
    #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #     )


    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
