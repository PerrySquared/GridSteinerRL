from gymnasium import spaces
from overflow import OverflowWithOverlap
from .utils.dataset_extractor import get_coords_dataset
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
import numpy as np
import random
import h5py
import copy
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

TERMINAL_CELL = 2
PATH_CELL = 1 

RENDER_EACH = 1
RESET_EACH = 1

TARGETS_TOTAL = 4

LOCAL_AREA_SIZE = 32

TEMP_GENERAL_OVERFLOW = np.zeros((1000,1000), dtype=np.float64)

random.seed(11)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, size=LOCAL_AREA_SIZE):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        self._target_locations_copy = []
        self._agent_location_copy = []
        self._target_locations = []
        self._agent_location = []
        self.found_instance_index = 0
        self.env_steps = 0
        self.env_swaps = 0
        self.f = h5py.File('./gym_examples/envs/utils/dataset.h5', 'r')
        
        render_mode = None
        
        # self.observation_space = spaces.Box(0, 1, shape=(96, 96), dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                # "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                # "target_locations": spaces.Box(0, 5, shape=(self.size, self.size), dtype=np.float64),
                "target_matrix": spaces.Box(0, 1, shape=(LOCAL_AREA_SIZE, LOCAL_AREA_SIZE), dtype=np.float64),
                "reference_overflow_matrix": spaces.Box(0, 1, shape=(LOCAL_AREA_SIZE, LOCAL_AREA_SIZE), dtype=np.float64),
                "target_list": spaces.Box(0, self.size, shape=(4,2), dtype=np.int64),
                "targets_left": spaces.Discrete(5),
                # "targets_relative_line": spaces.Discrete(5),
                # "targets_relative_general": spaces.Box(0, 1, shape=(4,), dtype=int),
                # add reference overflow matrix, i.e. cutout of the standard size from the general overflow 
            }
        )

        self.action_space = spaces.MultiDiscrete(np.array([4, 4]))
        
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
        self.iterations = 0
        self.total_footprint = 0
        self.initial_targets_amount = 1
        self.stagnate_counter = 0
        self.previous_actions = []
        # call for overflow class to create local overflow matrix
        self.Overflow = OverflowWithOverlap(TEMP_GENERAL_OVERFLOW, LOCAL_AREA_SIZE, LOCAL_AREA_SIZE)

        global RESET_EACH
        global TARGETS_TOTAL
        
        TASK_TARGETS = 4
        
        if self.env_steps % RESET_EACH == 0:    # get new random env every 100 envs
            self._target_locations_copy = np.full((4,2), -1)
            # !!! instead of random iterate over extracted nets in order, slowly building up general overflow matrix and save it when no more nets left (building up when condition at the end of step is satisfied)
            temp_target_locations_copy, self.net_name, self.insertion_coords, self.origin_shape, self.found_instance_index = get_coords_dataset(self.found_instance_index + 1, TASK_TARGETS, self.f)        
            
            TARGETS_TOTAL = len(temp_target_locations_copy)
            self._target_locations_copy[:TARGETS_TOTAL] = temp_target_locations_copy
            
            self.env_swaps += 1
            
            
        self.env_steps += 1
        
        self.Overflow.create_overflow_reference_matrix(self.insertion_coords)
        
        self._target_locations = copy.deepcopy(self._target_locations_copy) # copy to have an independent instance to interact with

        for _target_location in self._target_locations:
            if _target_location[0] != -1 and _target_location[1] != -1:
                self.Overflow.local_overflow_matrix[_target_location[0], _target_location[1]] = TERMINAL_CELL
            
        
        observation = self._get_obs()
        info = self._get_info()

        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            if self.render_mode == "human":
                self._render_frame()
            if self.render_mode == "rgb_array":
                self._render_frame_as_rgb_array()

        return observation, info
    
    def get_target_amount_sequence(self, env_swaps):

        if env_swaps%10 == 0:
            print("env_swaps ", env_swaps)
        
        if env_swaps <= 800:
            return 3
        elif env_swaps > 800 and env_swaps <= 1600:
            return 4
        elif env_swaps > 1600 and env_swaps <= 3500:
            return 5
        else:
            return 5

    def step(self, action):
        # check if exited
        if self.env_steps % RENDER_EACH == 0:
            if self.render_mode == "human" or self.render_mode == "rgb_array":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

        global TARGETS_TOTAL
        successful_path = False
        
        reward = 0
        terminated = False     
        truncated = False
        
        self.iterations += 1
        
        unsuccessful_move, step_overflow, normalized_step_overflow, path_length = self._move(action) # update the position
  
        if np.count_nonzero(self.Overflow.local_overflow_matrix == TERMINAL_CELL) == 0: # successful game over (no terminals left)
            terminated = True
            successful_path = True
            reward += 1 
            reward += TARGETS_TOTAL/5
            
        # avg overflow per cell on path      
        path_length = path_length if path_length > 0 else 1
        reward -= normalized_step_overflow / path_length

        if unsuccessful_move: # if picked action that has negative pair of coords
            reward -= 1
            truncated = True
        
        if self.iterations > 5: # quit if too many steps
            reward -= 1
            truncated = True
        
        # if both elements picked by action are the same to prevent just picking the terminals
        # if action[0] == action[1]: 
        #     reward -= 1

        # if connects same terminals as one of the previous actions in any order
        # if self.iterations > 1 and self.is_identical_to_previous(action, self.previous_actions):
        #     reward -= 1 
        #     print("ident_prev", reward)
        
        # if current action doesnt include one of the previous actions (to prevent not connected paths between two pairs of terminals) unless the first iteration
        if self.iterations > 1 and (not self.is_connected_to_previous(action, self.previous_actions) or self.is_identical_to_previous(action, self.previous_actions)): 
            reward -= 1

        # if self.iterations > 1 and self.is_connected_to_previous(action, self.previous_actions) and not self.is_identical_to_previous(action, self.previous_actions) and not action[0] == action[1]:
        #     reward += 0.8


        self.previous_actions.append(action)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.env_steps % RENDER_EACH == 0: # render only mod N env

            if self.render_mode == "human":
                self._render_frame()
            if self.render_mode == "rgb_array":
                self._render_frame_as_rgb_array()

        # 6. return game over and score

        # !!! if terminated or truncated add local overflow to general overflow (dependent on the amount of repeats for a single env, i.e. save best or latest out of RESET_EACH)
        if successful_path:
            global TEMP_GENERAL_OVERFLOW
            
            upto_row = min(self.insertion_coords[0] + LOCAL_AREA_SIZE, self.origin_shape[0])
            upto_column = min(self.insertion_coords[1] + LOCAL_AREA_SIZE, self.origin_shape[1])
            
            limit_row = self.origin_shape[0] - self.insertion_coords[0]
            limit_column = self.origin_shape[1] - self.insertion_coords[1]
            
            # maybe save only the most optimal local matrix, as of now it saves every correctly created one thus overloading the general matrix
            TEMP_GENERAL_OVERFLOW[self.insertion_coords[0]:upto_row, self.insertion_coords[1]:upto_column] += self.Overflow.local_overflow_matrix[0:limit_row, 0:limit_column] 

        reward /= TARGETS_TOTAL # divide the reward depending on the amount of targets in the task
        
        # print(reward, action)
        return observation, reward, terminated, truncated, info



    
    def _move(self, action):
        first_terminal = self._target_locations[action[0]]
        second_terminal = self._target_locations[action[1]]     

        if first_terminal[0] == -1 or first_terminal[1] == -1 or second_terminal[0] == -1 or second_terminal[1] == -1: # no moves between coords = -1 allowed
            return True, 1, 1, 1 # overall with give a negative reward later, i.e. -= 1/1
        else:
            step_overflow, normalized_step_overflow, path_length, path_matrix = self.Overflow.get_manhattan_path(first_terminal, second_terminal)
            return False, step_overflow, normalized_step_overflow, path_length



    def is_connected_to_previous(self, pair, previous_pairs):
        # Check if the pair is connected to any of the previous pairs
        for prev_pair in previous_pairs:
            # identical or differently ordered pairs dont count as connected to previous
            if  pair[0] != pair[1] and prev_pair [0] != prev_pair[1] and \
                ((pair[0] == prev_pair[0] and pair[1] != prev_pair[1]) or \
                (pair[1] == prev_pair[0] and pair[0] != prev_pair[1]) or \
                (pair[0] == prev_pair[1] and pair[1] != prev_pair[0]) or \
                (pair[1] == prev_pair[1] and pair[0] != prev_pair[0])):
                    return True
        return False

    def is_identical_to_previous(self, pair, previous_pairs):
        # Check if the pair is identical to any of the previous ones in any order
        for prev_pair in previous_pairs:
            if (pair[0] == prev_pair[0] and pair[1] == prev_pair[1]) or (pair[0] == prev_pair[1] and pair[1] == prev_pair[0]):
                return True
        return False
    
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


    """observation and info methods"""

    def _get_obs(self):
        # print(self._agent_location,"\n", self.matrix_with_targets,"==============")
        # print({"targets_relative_line": self.check_for_target_line(), "targets_relative_general": self.check_for_target_general()})
        output_shape = (LOCAL_AREA_SIZE, LOCAL_AREA_SIZE)
        padding = ((3, 3), (3, 3))  # Padding of 2 rows and 2 columns on each side
        
        output_array = self.resize(np.divide(self.Overflow.local_overflow_matrix, 2), output_shape, 0)
        output_overflow = self.resize(self.Overflow.overflow_reference_matrix, output_shape, 0)
        
        output_overflow_min = output_overflow.min()
        output_overflow_max = output_overflow.max()
        if output_overflow_min == output_overflow_max:
            normalized_output_overflow = output_overflow * 0
        else:
            normalized_output_overflow = (output_overflow - output_overflow_min) / (output_overflow_max - output_overflow_min)
        
        # print({
        #     # "agent": self._agent_location,
        #     # "target_locations": self.get_matrix_with_targets(),
        #     "target_matrix": output_array,
        #     "reference_overflow_matrix": normalized_output_overflow,
        #     "target_list": np.array(self._target_locations, dtype=np.int64),
        #     "targets_left": np.count_nonzero(self.Overflow.local_overflow_matrix == TERMINAL_CELL),
        #     # "targets_relative_line": self.check_for_target_line(),
        #     # "targets_relative_general": self.check_for_target_general(),
        #     })
        
        return {
            # "agent": self._agent_location,
            # "target_locations": self.get_matrix_with_targets(),
            "target_matrix": output_array,
            "reference_overflow_matrix": normalized_output_overflow,
            "target_list": np.array(self._target_locations, dtype=np.int64),
            "targets_left": np.count_nonzero(self.Overflow.local_overflow_matrix == TERMINAL_CELL),
            # "targets_relative_line": self.check_for_target_line(),
            # "targets_relative_general": self.check_for_target_general(),
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
            # "target_matrix": self.Overflow.local_overflow_matrix,
            #  "reference_overflow_matrix": self.Overflow.overflow_reference_matrix
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
        for i in range(len(self.Overflow.local_overflow_matrix)):
            for j in range(len(self.Overflow.local_overflow_matrix)):
                color = None
                if self.Overflow.local_overflow_matrix[i, j] == PATH_CELL:
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
    #     for i in range(len(self.Overflow.local_overflow_matrix)):
    #         for j in range(len(self.Overflow.local_overflow_matrix)):
    #             color = None
    #             if self.Overflow.local_overflow_matrix[i, j] == PATH_CELL:
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
