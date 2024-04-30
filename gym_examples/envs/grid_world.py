import gymnasium as gym
from gymnasium import spaces
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

TRACE_CELL = 1
INTERSECTION_CELL = 2
TERMINAL_CELL = 3
PATH_CELL = 4
AGENT = 5

RENDER_EACH = 100000
RESET_EACH = 2000

random.seed(11)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, size=32):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        self._target_locations_copy = []
        self._agent_location_copy = []
        self._target_locations = []
        self._agent_location = []
        self.random_element = 0
        self.env_steps = 0
        
        # self.observation_space = spaces.Box(0, 1, shape=(96, 96), dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                # "target_locations": spaces.Box(0, 5, shape=(self.size, self.size), dtype=np.float64),
                "target_matrix": spaces.Box(0, 1, shape=(96, 96), dtype=np.float64),
                "targets_relative_line": spaces.Discrete(5),
                "targets_relative_general": spaces.Box(0, 1, shape=(4,), dtype=int),
                "targets_left": spaces.Discrete(5)
            }
        )
        
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, -1]), # up
            1: np.array([0, 1]), # down
            2: np.array([-1, 0]), # left
            3: np.array([1, 0]), # right
        }
        
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
        self.matrix = np.zeros((self.size,self.size), dtype=int)
        self.iterations = 1
        self.total_footprint = 0
        self.previous_position = np.array([-1, -1])
        self.initial_targets_amount = 1
        self.stagnate_counter = 0
        
        if self.env_steps % RESET_EACH == 0:    # get new random env every 100 envs
            self._target_locations_copy = self.get_target_locations()
            self._agent_location_copy = random.choice(self._target_locations_copy)   
            
        self._target_locations = copy.deepcopy(self._target_locations_copy)
        self._agent_location = copy.deepcopy(self._agent_location_copy)

        self.env_steps += 1

        for _target_location in self._target_locations:
            self.place_support_nodes(self._target_locations)
            self.matrix[_target_location[0], _target_location[1]] = TERMINAL_CELL
            

        self.check_target_location() # remove first target that agent is on top of
        self.initial_targets_amount = len(self._target_locations) # get the amount of targets left
        
        """marking current agent position in a matrix and saving its position as previous one to use later"""
        self.matrix[self._agent_location[0]][self._agent_location[1]] = AGENT
        self.previous_position = copy.deepcopy(self._agent_location)
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            if self.render_mode == "human":
                self._render_frame()
            if self.render_mode == "rgb_array":
                self._render_frame_as_rgb_array()


        return observation, info
    
    def get_target_locations(self):
        _target_locations = []
        for i in range(random.randint(2,5)):
            _target_locations.append(np.array([random.randint(0, self.size - 1), random.randint(0, self.size - 1)]))
        return _target_locations
    
    def place_support_nodes(self, terminals):
        self.make_path([0, 0], [0, self.size - 1], [0, 1])
        self.make_path([0, 0], [self.size - 1, 0], [1, 0])
        self.make_path([0, self.size - 1], [self.size - 1, self.size - 1], [1, 0])
        self.make_path([self.size - 1, 0], [self.size - 1, self.size - 1], [0, 1])

        for t__ in terminals:
            self.make_path([t__[0],0], [t__[0], self.size - 1], [0, 1])
            self.make_path([0,t__[1]], [self.size - 1, t__[1]], [1, 0])

    def make_path(self, iterator, end, step):
        if(self.matrix[iterator[0], iterator[1]] == INTERSECTION_CELL or self.matrix[iterator[0], iterator[1]] == TERMINAL_CELL):
            return
        
        while(iterator <= end):
            if(self.matrix[iterator[0], iterator[1]] == TRACE_CELL):
                self.matrix[iterator[0], iterator[1]] = INTERSECTION_CELL
            elif(self.matrix[iterator[0],iterator[1]] != TERMINAL_CELL):
                self.matrix[iterator[0], iterator[1]] = TRACE_CELL


            iterator[0] += step[0]
            iterator[1] += step[1]
            

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
        not_moved = False
        terminated = False     
        truncated = False
        
        not_moved = self._move(action) # update the position 
         
        reward, terminated, truncated = self.game_over_check()
            
        # print(not_moved)
        if not_moved:
            reward = -1
            
        
        observation = self._get_obs()
        info = self._get_info()
        # print("observed\n")
        
        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            print(reward)
            if self.render_mode == "human":
                self._render_frame()
            if self.render_mode == "rgb_array":
                self._render_frame_as_rgb_array()
            
        # 6. return game over and score
        return observation, reward, terminated, truncated, info

    
    def _move(self, action):       
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        _agent_location = copy.deepcopy(self._agent_location)
        _agent_location = np.clip(_agent_location + direction, 0, self.size - 1)

        if self.matrix[_agent_location[0], _agent_location[1]] != 0:
            if self.matrix[_agent_location[0], _agent_location[1]] == TRACE_CELL or self.matrix[_agent_location[0], _agent_location[1]] == PATH_CELL:
                while self.matrix[_agent_location[0], _agent_location[1]] == TRACE_CELL or self.matrix[_agent_location[0], _agent_location[1]] == PATH_CELL:
                    self.matrix[_agent_location[0], _agent_location[1]] = PATH_CELL
                    _agent_location = np.clip(_agent_location + direction, 0, self.size - 1)
        else:
            return False, True # in case of unexpected movement (if it's even possible) - will shut down the proccess with an error
        
        """three lines below are needed to create context in the matrix about agent location,
        i.e. making current position of an agent a 5 and changing the previous one into 
        intersection cell remove if not needed along with the one in reset"""
        
        self.matrix[self.previous_position[0]][self.previous_position[1]] = INTERSECTION_CELL 
        self.matrix[_agent_location[0]][_agent_location[1]] = AGENT # agent marked AFTER intersection in case it stays in the same place
        not_moved = np.array_equal(_agent_location, self.previous_position) # check if agent hasn't moved (due to border clipping)
        self.previous_position = copy.deepcopy(_agent_location)
        
        self._agent_location = _agent_location
        
        return not_moved

    def game_over_check(self):
        # print("targets: ", self._target_locations)
        reward = 0
        terminated = False
        truncated = False
        
        footprint = (self.matrix == PATH_CELL).sum() # how many path cells are there
            
        # comparing footprint after current step with the previous one (might want to add && not_moved == False to consider intersections next to each other)
        if footprint > self.total_footprint: 
            reward -= (1 / self.size) * (footprint - self.total_footprint) # roughly 1/30th of the max footprint in a single step so max negative per step is 0.99 - can be calculated as 1/self.size if size is variable
            self.total_footprint = copy.deepcopy(footprint)
            self.stagnate_counter = 0
        else:
            self.stagnate_counter += 1
            if self.stagnate_counter > 5:
                reward = -0.9 # punish walking along the same path endlessly
        
        # add reward for collecting _target_locations
        if self.check_target_location():
            reward += 0.9 # / self.initial_targets_amount # lessen the reward for collecting single target (dependant on the target amount)
        else:
            reward -= 0.03 # stimulate searching for targets
            
        if len(self._target_locations) == 0: # successful game over
            terminated = True
            reward = 1 # was uncommented
            
        if self.iterations > 30: # terminated due to excessive amount of steps
            # print("trunc")
            reward = -1
            truncated = True

        return reward, terminated, truncated # score is less with each step
    

    def check_target_location(self):
        for index in range(len(self._target_locations)):
            if np.array_equal(self._agent_location, self._target_locations[index]):
                self.matrix[self._agent_location[0], self._agent_location[1]] = INTERSECTION_CELL
                del self._target_locations[index]
                return True
        return False
    
        # print(x,y)
        # print(np.transpose(self.matrix))
    
    """extra observation space methods"""
    
    def get_matrix_with_targets(self):
        matrix = np.zeros((self.size, self.size))
        for target in self._target_locations:
            matrix[target[0], target[1]] = TERMINAL_CELL
            
        return matrix

    def check_for_target_general(self):
        target_relative_positions = np.array([0, 0, 0, 0]) # top, bottom, left, right
        for x in range(self.size):
            for y in range(self.size):
                if self.matrix[x][y] == TERMINAL_CELL and y < self._agent_location[1]:
                    target_relative_positions[0] = 1
                if self.matrix[x][y] == TERMINAL_CELL and y > self._agent_location[1]:
                    target_relative_positions[1] = 1
        for y in range(self.size):
            for x in range(self.size):
                if self.matrix[x][y] == TERMINAL_CELL and x < self._agent_location[0]:
                    target_relative_positions[2] = 1
                if self.matrix[x][y] == TERMINAL_CELL and x > self._agent_location[0]:
                    target_relative_positions[3] = 1
                    
        return target_relative_positions
    
    def check_for_target_line(self):
        for y in range(self.size):
            if self.matrix[self._agent_location[0]][y] == TERMINAL_CELL and y < self._agent_location[1]:
                return 0 # top
            if self.matrix[self._agent_location[0]][y] == TERMINAL_CELL and y > self._agent_location[1]:
                return 1 # bottom
        for x in range(self.size):
            if self.matrix[x][self._agent_location[1]] == TERMINAL_CELL and x < self._agent_location[0]:
                return 2 # left
            if self.matrix[x][self._agent_location[1]] == TERMINAL_CELL and x > self._agent_location[0]:
                return 3 # right
            
        return 4 # none
    

    """observation and info methods"""

    def _get_obs(self):
        # print(self._agent_location,"\n", self.matrix,"==============")
        # print({"targets_relative_line": self.check_for_target_line(), "targets_relative_general": self.check_for_target_general()})
        output_shape = (96, 96)
        padding = ((8, 8), (8, 8))  # Padding of 2 rows and 2 columns on each side
        
        output_array = self.resize(np.divide(self.matrix, 5), output_shape, padding)
        # return output_array
        
        return {
            "agent": self._agent_location,
            # "target_locations": self.get_matrix_with_targets(),
            "target_matrix": output_array,
            "targets_relative_line": self.check_for_target_line(),
            "targets_relative_general": self.check_for_target_general(),
            "targets_left": len(self._target_locations)
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
            "agent": self._agent_location,
            "target_matrix": self.matrix
            }

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
                if self.matrix[i, j] == TRACE_CELL:
                    color = (100, 0, 0)
                elif self.matrix[i, j] == PATH_CELL:
                    color = (0, 0, 150)
                elif self.matrix[i, j] == INTERSECTION_CELL:
                    color = (0, 255, 0)
                elif self.matrix[i, j] == AGENT:
                    color = (0, 100, 0)
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

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
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

    import numpy as np


    def _render_frame_as_rgb_array(self): # not working yet
        
        if self.window is None and self.render_mode == "rgb_array":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "rgb_array":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels

        # Draw the target
        for _target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * _target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        font = pygame.font.Font(None, 24)  # Choose a font and size
        text_surface = font.render(str(self.env_steps), True, (0, 0, 255))  # Render the text
        canvas.blit(text_surface, (10, 10))  # Blit the text onto the canvas

        # Draw the agent and other elements
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                color = None
                if self.matrix[i, j] == TRACE_CELL:
                    color = (100, 0, 0)
                elif self.matrix[i, j] == PATH_CELL:
                    color = (0, 0, 150)
                elif self.matrix[i, j] == INTERSECTION_CELL:
                    color = (0, 255, 0)
                elif self.matrix[i, j] == AGENT:
                    color = (0, 100, 0)
                if color:
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            int(pix_square_size / 3),
        )

        # Add gridlines
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

        self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
