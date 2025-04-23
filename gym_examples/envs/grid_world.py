from gymnasium import spaces
from overflow import OverflowWithOverlap3D  # Changed to 3D version
from .utils.dataset_extractor import get_coords_dataset, get_coords_dataset_3d  # Assuming this is updated for 3D
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
import numpy as np
import random
import h5py
import copy
import math
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.set_printoptions(threshold=sys.maxsize)

TERMINAL_CELL = 2
PATH_CELL = 1 

RENDER_EACH = 10000000000000
RESET_EACH = 1

# Updated for 3D
LOCAL_AREA_SIZE = 32  # Reduced to make 3D visualization more manageable
LAYERS = 6

# Initialize with 3D matrix
TEMP_GENERAL_OVERFLOW = np.zeros((1000, 1000, LAYERS), dtype=np.float64)
BACKUP_LOCAL_OVERFLOW = np.zeros((32, 32, LAYERS), dtype=np.float64)
# random.seed(11)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "plotly"], "render_fps": 1}

    def __init__(self, render_mode=None, size=LOCAL_AREA_SIZE, ppo_aproach=True, pins=2, 
                 general_overflow_matrix=None, file = None):
        self.size = size  # The size of the cubic grid
        self.window_size = 512  # The size of the PyGame window
        
        # Store the parameters passed via env_kwargs
        self.ppo_aproach = ppo_aproach
        self.pins = pins
        self.general_overflow_matrix = general_overflow_matrix if general_overflow_matrix is not None else TEMP_GENERAL_OVERFLOW
        self.f = file
        
        self._target_locations_copy = np.zeros((LOCAL_AREA_SIZE, LOCAL_AREA_SIZE, LAYERS), dtype=np.float64)
        self._target_locations = np.zeros((LOCAL_AREA_SIZE, LOCAL_AREA_SIZE, LAYERS), dtype=np.float64)
        self._target_locations_used = np.zeros((self.pins, 3), dtype=np.int64)
        
        self._agent_location_copy = []
        self._agent_location = []
        self.found_instance_index = 0
        self.env_steps = 0
        self.env_swaps = 0
        
        
        render_mode = "plotly"
        
        # Updated observation space for 3D
        self.observation_space = spaces.Dict(
            {
                "target_matrix": spaces.Box(0, 1, shape=(LOCAL_AREA_SIZE, LOCAL_AREA_SIZE, LAYERS), dtype=np.float64),
                "reference_overflow_matrix": spaces.Box(0, 1, shape=(LOCAL_AREA_SIZE, LOCAL_AREA_SIZE, LAYERS), dtype=np.float64),
                "target_list": spaces.Box(0, self.size, shape=(self.pins, 3), dtype=np.int64), 
                "target_list_used": spaces.Box(0, 1, shape=(self.pins, 3), dtype=np.int64),
            }
        )

        self.action_space = spaces.MultiDiscrete(np.array([self.pins, self.pins, 2, LAYERS])) # 2 is for which rectilinear path to take in the selected layer
        
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
        self.successful_path = False
        self.failed_path = False
        self.total_footprint = 0
        self.initial_targets_amount = 1
        self.stagnate_counter = 0
        self.previous_actions = []
        

        global RESET_EACH
        
        # Call for 3D overflow class to create local overflow matrix
        self.Overflow = OverflowWithOverlap3D(self.general_overflow_matrix, LOCAL_AREA_SIZE, LOCAL_AREA_SIZE, LAYERS)
        
        self._target_locations_used = np.zeros((self.pins, 3), dtype=np.int64)
        
        # print("-reset-")
        
        if self.env_steps % RESET_EACH == 0:    # get new env every N envs
            # for layer in range(LAYERS):
            #     max_val = random.randint(1, 10)
            #     min_val =  random.randint(0, max_val - 1)
            #     self.Overflow.general_overflow_matrix[:,:,layer] = np.random.randint(min_val * 100, max_val * 100, size=(1000, 1000))
            
            self._target_locations_copy = np.full((self.pins, 3), -1)  # Updated to 3D coordinates
            # Get 3D coordinates from the dataset
            temp_target_locations_copy, self.net_name, self.insertion_coords, self.origin_shape, self.found_instance_index = get_coords_dataset_3d(self.found_instance_index + 1, self.pins, self.f) 
            
            if not self.net_name:
                observation = self._get_obs()
                info = self._get_info()
                return observation, info
                
            self.pins = len(temp_target_locations_copy)
            self._target_locations_copy[:self.pins] = temp_target_locations_copy
            
            self.env_swaps += 1
            
        self.env_steps += 1
        
        self.Overflow.create_overflow_reference_matrix(self.insertion_coords)
        
        self._target_locations = copy.deepcopy(self._target_locations_copy) # copy to have an independent instance to interact with

        for _target_location in self._target_locations:
            if _target_location[0] != -1 and _target_location[1] != -1 and _target_location[2] != -1:
                # print(_target_location[0], _target_location[1], _target_location[2])
                self.Overflow.local_overflow_matrix[_target_location[0], _target_location[1], _target_location[2]] = TERMINAL_CELL
        
        global BACKUP_LOCAL_OVERFLOW
        BACKUP_LOCAL_OVERFLOW = copy.deepcopy(self.Overflow.local_overflow_matrix)
        
        observation = self._get_obs()
        info = self._get_info()
        # print(info)
        
        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            if self.render_mode == "human":
                self._render_frame()
            elif self.render_mode == "plotly":
                self._render_plotly()

        return observation, info
    
    # def get_target_amount_sequence(self, env_swaps):
    #     if env_swaps % 10 == 0:
    #         print("env_swaps ", env_swaps)
        
    #     if env_swaps <= 800:
    #         return 3
    #     elif env_swaps > 800 and env_swaps <= 1600:
    #         return 4
    #     elif env_swaps > 1600 and env_swaps <= 3500:
    #         return 5
    #     else:
    #         return 5

    def step(self, action):
        
        if not self.net_name:
            observation = self._get_obs()
            info = self._get_info()
            return observation, 0, False, True, info
              
        reward = 0
        terminated = False     
        truncated = False
        
        self.iterations += 1
        
        # print(action)
        
        self._target_locations_used[action[0:2]] = 1
        # print(self._target_locations)
        unsuccessful_move, step_overflow, normalized_step_overflow, path_length = self._move(action) # update the position
        # print(normalized_step_overflow)
        
        if np.count_nonzero(self.Overflow.local_overflow_matrix == TERMINAL_CELL) == 0: # successful game over (no terminals left)
            terminated = True
            self.successful_path = True
            reward += 1
            # reward += self.pins/5
        
        # print("unsuc move ", unsuccessful_move)
        if unsuccessful_move: # if picked action that has negative pair of coords
            reward -= 1
            truncated = True         
        
        # print("iter ", self.iterations)
        if self.iterations > 5: # quit if too many steps
            reward -= 1
            truncated = True
            self.failed_path = True
            
            # backup solution with a switch here
        
        # avg overflow per cell on path, if 0 then path length with a coef
        if normalized_step_overflow > 0:
            reward -= normalized_step_overflow * 1.75
        else:
            reward -= path_length * 0.04

        # if both elements picked by action are the same to prevent just picking the terminals
        if action[0] == action[1]: 
            reward -= 1
        
        # the lower the layer the higher the neg reward
        # reward -= action[3] * 0.01
        
        # if changes layers after first move then neg reward
        # if self.iterations > 1:
        #     if self.previous_actions[0][3] != action[3]:
        #         reward -= 0.1
            # for prev_action in self.previous_actions:
            #     if prev_action[3] == action[3]:
            #         reward += 0.1
            #     else:
            #         reward -= 0.1
        
        # print("identical to previous ", self.iterations > 1 and self.is_identical_to_previous(action, self.previous_actions))
        # if connects same terminals as before
        if self.iterations > 1 and self.is_identical_to_previous(action, self.previous_actions):
            reward -= 1 # try * self.iterations
        
        # print("not connected ",  self.iterations > 1 and not self.is_connected_to_previous(action, self.previous_actions))
        # if current action doesnt include one of the previous actions (to prevent not connected paths between two pairs of terminals) unless the first iteration
        if self.iterations > 1 and not self.is_connected_to_previous(action, self.previous_actions): # if not first iteration and current action contains only one element from the previous
            reward -= 0.7

        # print("connected ", self.iterations > 1 and self.is_connected_to_previous(action, self.previous_actions) and not self.is_identical_to_previous(action, self.previous_actions) and not action[0] == action[1])        
        if self.iterations > 1 and self.is_connected_to_previous(action, self.previous_actions) and not self.is_identical_to_previous(action, self.previous_actions) and not action[0] == action[1]:
            reward += 0.5
        
        # print('rew2=', reward)
            
        self.previous_actions.append(action)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.env_steps % RENDER_EACH == 0: # render only mod N env
            if self.render_mode == "human":
                self._render_frame()
            elif self.render_mode == "plotly":
                self._render_plotly()
                if(terminated or truncated):
                    input("Press the <ENTER> key to continue...")

        # If terminated or truncated, add local overflow to general overflow
        if self.env_steps % 1 == 0 and (terminated or truncated):
        # if terminated:
            
            # Update for 3D coordinates
            upto_row = min(self.insertion_coords[0] + LOCAL_AREA_SIZE, self.origin_shape[0])
            upto_column = min(self.insertion_coords[1] + LOCAL_AREA_SIZE, self.origin_shape[1])
            upto_z = min(self.insertion_coords[2] + LAYERS, self.origin_shape[2])
            
            limit_row = self.origin_shape[0] - self.insertion_coords[0]
            limit_column = self.origin_shape[1] - self.insertion_coords[1]
            limit_z = self.origin_shape[2] - self.insertion_coords[2]
            # print("===================")

            self.general_overflow_matrix[
                self.insertion_coords[0]:upto_row, 
                self.insertion_coords[1]:upto_column,
                self.insertion_coords[2]:upto_z
            ] += self.Overflow.local_overflow_matrix[0:limit_row, 0:limit_column, 0:limit_z]
            
        # Scale the reward from [-2.76, 0.7] to [-1, 1] using the formula
        # new_value = new_min + (old_value - old_min) * (new_max - new_min) / (old_max - old_min)
        # scaled_reward = -1 + ((reward + 3.1) * 2)/4.6
        # Clip to ensure the scaled reward stays within [-1, 1]
        # scaled_reward = np.clip(scaled_reward, -1, 1)
        
        return observation, reward, terminated, truncated, info

    def _move(self, action):
        first_terminal = self._target_locations[action[0]]
        second_terminal = self._target_locations[action[1]]
        path_type = action[2]
        layer = action[3]

        if (first_terminal[0] == -1 or first_terminal[1] == -1 or first_terminal[2] == -1 or 
            second_terminal[0] == -1 or second_terminal[1] == -1 or second_terminal[2] == -1): 
            # No moves between coords = -1 allowed
            return True, 1, 1, 1 # overall with give a negative reward later, i.e. -= 1/1
        else:
            step_overflow, normalized_step_overflow, path_length, path_matrix = self.Overflow.get_manhattan_path(first_terminal, second_terminal, path_type, layer)
            return False, step_overflow, normalized_step_overflow, path_length

    def is_identical_to_previous(self, pair, previous_pairs):
        pair_pin_ids = pair[0:2] 
        # Check if the pair's coordinates are identical to any of the previous ones in any order
        for prev_pair in previous_pairs:
            prev_pair_pin_ids = prev_pair[0:2]
            
            if ((pair_pin_ids[0] == prev_pair_pin_ids[0] and pair_pin_ids[1] == prev_pair_pin_ids[1])
            or (pair_pin_ids[0] == prev_pair_pin_ids[1] and pair_pin_ids[1] == prev_pair_pin_ids[0])):
                return True
            
        return False
    
    def is_connected_to_previous(self, pair, previous_pairs):
        pair_pin_ids = pair[0:2] 
        # Check if the pair is connected to any of the previous pairs of selected actions
        for prev_pair in previous_pairs:
            prev_pair_pin_ids = prev_pair[0:2]
            
            if (pair_pin_ids[0] == prev_pair_pin_ids[0] or pair_pin_ids[1] == prev_pair_pin_ids[0] 
            or pair_pin_ids[0] == prev_pair_pin_ids[1] or pair_pin_ids[1] == prev_pair_pin_ids[1]):
                return True
            
        return False

    
    
    def _get_obs(self):
        # For 3D, update shapes and padding
        output_shape = (LOCAL_AREA_SIZE, LOCAL_AREA_SIZE, LAYERS)
        padding = ((3, 3), (3, 3), (3, 3))  # Padding in all three dimensions
        
        output_array = self.resize_3d(np.divide(self.Overflow.local_overflow_matrix, 2), output_shape)
        output_overflow = self.resize_3d(self.Overflow.overflow_reference_matrix, output_shape)
        
        output_overflow_min = output_overflow.min()
        output_overflow_max = output_overflow.max()
        denominator = max(output_overflow_max - output_overflow_min, 1)

        normalized_output_overflow = (output_overflow - output_overflow_min) / denominator
            
        return {
            "target_matrix": output_array,
            "reference_overflow_matrix": normalized_output_overflow,
            "target_list": np.array(self._target_locations, dtype=np.int64),
            "target_list_used": np.array(self._target_locations_used, dtype=np.int64),
        }

    def resize_3d(self, array, output_shape, padding=0):
        # Following your exact approach but for 3D
        old_shape = array.shape
        
        # Calculate scale factors for all three dimensions
        rows_scale = output_shape[0] // old_shape[0]
        cols_scale = output_shape[1] // old_shape[1]
        layers_scale = output_shape[2] // old_shape[2]
        
        # Repeat values in all three dimensions
        repeated_rows = np.repeat(array, rows_scale, axis=0)
        repeated_cols = np.repeat(repeated_rows, cols_scale, axis=1)
        repeated_layers = np.repeat(repeated_cols, layers_scale, axis=2)
        
        # Trim to desired shape
        resized_array = repeated_layers[:output_shape[0], :output_shape[1], :output_shape[2]]
        
        # padded_array = np.pad(resized_array, padding, mode='constant', constant_values=0)
        
        return resized_array

    def _get_info(self):
        return {
            'insertion_coords': self.insertion_coords,
            'origin_shape': self.origin_shape,
            'local_overflow_matrix': BACKUP_LOCAL_OVERFLOW,
            'reference_overflow_matrix': self.Overflow.overflow_reference_matrix
        }


    def _render_frame(self):
        """Render the current state of the environment using Pygame 3D visualization"""
        pass

    # Optional: Add a Plotly-based 3D visualization method
    def _render_plotly(self, colormap=None, opacity=1.0, 
                        show_axes=True, title='3D Matrix Visualization', 
                        marker_size_path=6, marker_size_filler=3, z_scale=0.3, skip_zeros=True):
        """
        Visualize a 3D matrix/tensor using Plotly.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            The 3D matrix to visualize (shape: [x_size, y_size, z_size])
        method : str, optional
            Visualization method: 'volume', 'isosurface', or 'scatter'
        colorscale : str, optional
            Colorscale to use for the visualization
        opacity : float, optional
            Opacity for the 3D visualization (0.0 to 1.0)
        show_axes : bool, optional
            Whether to show the axes labels and ticks
        title : str, optional
            Title for the plot
        show_colorbar : bool, optional
            Whether to show the colorbar
        surface_count : int, optional
            Number of iso-surfaces (for volume and isosurface methods)
        marker_size : float, optional
            Size of markers (for scatter method)
        """
        
        local_matrix = self.Overflow.local_overflow_matrix
        reference_matrix = self.Overflow.overflow_reference_matrix
        
        # Check input
        if not isinstance(local_matrix, np.ndarray):
            try:
                local_matrix = np.array(local_matrix)
            except:
                raise ValueError("Input matrix must be convertible to a numpy array")
        
        if local_matrix.ndim != 3:
            raise ValueError(f"Expected a 3D matrix, got {local_matrix.ndim}D array")
        
        # Default colormap for values 1, 2 (0 is transparent)
        if colormap is None:
            colormap = {1: 'rgba(255, 0, 0, 0.7)', 2: 'rgba(0, 255, 0, 0.7)'}
        
        # Create figure
        fig = go.Figure()
        
        # Get dimensions
        x_size, y_size, z_size = reference_matrix.shape
        
        # STEP 1: Create coordinates for EVERY cell in the 3D grid
        X, Y, Z = np.mgrid[0:x_size, 0:y_size, 0:z_size]
        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
        Z_scaled_flat = Z_flat * z_scale
        
        # Get reference values for all points
        ref_values_flat = reference_matrix.flatten()
        
        # Filter out zeros if skip_zeros is True
        if skip_zeros:
            display_mask = ref_values_flat != 0
        else:
            display_mask = np.ones_like(ref_values_flat, dtype=bool)
        
        # Add heatmap dots for ALL reference matrix cells (with optional zero filtering)
        # Use only the points that passed the display mask
        X_display = X_flat[display_mask]
        Y_display = Y_flat[display_mask]
        Z_display = Z_scaled_flat[display_mask]
        values_display = ref_values_flat[display_mask]
        
        fig.add_trace(go.Scatter3d(
            x=X_display,
            y=Y_display,
            z=Z_display,
            mode='markers',
            marker=dict(
                size=marker_size_filler,
                color=values_display,  # Use the reference values for color
                colorscale='Viridis',  # Use a heat map color scale
                opacity=0.5,
                symbol='circle',
                colorbar=dict(
                    title="Reference Values",
                    thickness=20,
                    len=0.75
                )
            ),
            hovertemplate='Value: %{text}<extra></extra>',
            text=values_display,
            name='Reference Matrix'
        ))
        
        # STEP 2: Overlay the path visualization with special colors (from local_matrix)
        local_flat = local_matrix.flatten()
        
        # Only add path points for values in the colormap (1 and 2)
        for val in [1, 2]:
            path_mask = local_flat == val
            if np.any(path_mask):
                # Get the same reference values for hover text
                path_ref_values = ref_values_flat[path_mask]
                
                fig.add_trace(go.Scatter3d(
                    x=X_flat[path_mask],
                    y=Y_flat[path_mask],
                    z=Z_scaled_flat[path_mask],
                    mode='markers',
                    marker=dict(
                        size=marker_size_path,
                        color=colormap[val],
                        opacity=opacity,
                        symbol='circle'
                    ),
                    hovertemplate='Value: %{text}<extra></extra>',
                    text=path_ref_values,
                    name=f'Path Value {val}'
                ))
        
        # Update layout
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
        
        axis_settings = dict(
            visible=show_axes,
            showgrid=show_axes,
            showticklabels=show_axes,
            showspikes=False,
            title=''
        )
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(**axis_settings, range=[-1, x_size]),
                yaxis=dict(**axis_settings, range=[-1, y_size]),
                zaxis=dict(**axis_settings, range=[-1, z_size * z_scale]),  # Scale z range accordingly
                aspectmode='manual',  # Allow different scales for axes
                aspectratio=dict(x=1, y=1, z=z_scale)  # Set aspect ratio
            ),
            scene_camera=camera,
            margin=dict(r=10, l=10, b=10, t=50 if title else 10),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # Display the figure
        fig.show()
