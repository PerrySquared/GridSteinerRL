import gym
from gym import spaces
import pygame
import numpy as np
import random


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1000}

    def __init__(self, render_mode="human", size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        self._target_locations_copy = None
        self._agent_location_copy = None
        self.random_element = 0
        self.env_steps = 0
        self.previous_position = np.array([-1, -1])
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, 1, shape=(size*size,), dtype=np.float64),
                "targets_relative_line": spaces.Discrete(5),
                "targets_relative_general": spaces.Box(0, 1, shape=(4,), dtype=int)
            }
        )
        
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, -1]), # up
            1: np.array([0, 1]), # down
            2: np.array([-1, 0]), # left
            3: np.array([1, 0]), # right
        }
        
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
        self.matrix = np.zeros((self.size,self.size))
        self.steps_counter = 1
        self.iterations = 1
        
        if self.env_steps % 200 == 0:    
            self._target_locations_copy = self.get_target_locations()
            self.random_element = random.randint(0, len(self._target_locations_copy) - 1)
                
            self._agent_location_copy = self._target_locations_copy[self.random_element]        
            del self._target_locations_copy[self.random_element]

        self.env_steps += 1
        print(self.env_steps)
        
        self._target_locations = self._target_locations_copy[:]
        self._agent_location = self._agent_location_copy[:]
        
        self.matrix[self._agent_location[0]][self._agent_location[1]] = 1 # make starting _target_location a part of the path
        
        for _target_location in self._target_locations:
            self.matrix[_target_location[0], _target_location[1]] = 0.5

        
        # print(np.transpose(self.matrix))
        # print(self.position)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def get_target_locations(self):
        _target_locations = []
        for i in range(random.randint(2,5)):
            _target_locations.append(np.array([random.randint(0, self.size - 1), random.randint(0, self.size - 1)]))
        # print(_target_locations)
        return _target_locations
    
    
    
    def _move(self, action):       
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # print("action: ", action)
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        x = self._agent_location[0]
        y = self._agent_location[1]

        if self.matrix[x][y] == 0 or self.matrix[x][y] == 0.5: # check if the next cell is already used by a wire from current net (in a more complex situation should be checked against an array of already used positions)
            self.steps_counter += 1 # if not add 1 to the total wire length
            self.matrix[x][y] = 1



    def check_target_location(self):
        for index in range(len(self._target_locations)):
            if np.array_equal(self._agent_location, self._target_locations[index]):
                del self._target_locations[index]
                return True
        return False
    
        # print(x,y)
        # print(np.transpose(self.matrix))

    def check_for_target_general(self):
        target_relative_positions = np.array([0, 0, 0, 0]) # top, bottom, left, right
        for x in range(10):
            for y in range(10):
                if self.matrix[x][y] == 0.5 and y < self._agent_location[1]:
                    target_relative_positions[0] = 1
                if self.matrix[x][y] == 0.5 and y > self._agent_location[1]:
                    target_relative_positions[1] = 1
        for y in range(10):
            for x in range(10):
                if self.matrix[x][y] == 0.5 and x < self._agent_location[0]:
                    target_relative_positions[2] = 1
                if self.matrix[x][y] == 0.5 and x > self._agent_location[0]:
                    target_relative_positions[3] = 1
                    
        return target_relative_positions
    
    def check_for_target_line(self):
        for y in range(10):
            if self.matrix[self._agent_location[0]][y] == 0.5 and y < self._agent_location[1]:
                return 0 # top
            if self.matrix[self._agent_location[0]][y] == 0.5 and y > self._agent_location[1]:
                return 1 # bottom
        for x in range(10):
            if self.matrix[x][self._agent_location[1]] == 0.5 and x < self._agent_location[0]:
                return 2 # left
            if self.matrix[x][self._agent_location[1]] == 0.5 and x > self._agent_location[0]:
                return 3 # right
            
        return 4
    
    
    
    def game_over_check(self):
        # print("targets: ", self._target_locations)
        reward = 0
        game_over = False
        self.iterations += 1
        
        if self.steps_counter > 50 or self.iterations > 100:
            game_over = True
            reward = -1

        if len(self._target_locations) == 0:
            game_over = True
            reward = 2

        if np.array_equal(self.previous_position, self._agent_location):
            reward -= 0.5
        self.previous_position = self._agent_location[:]

        return reward - 0.005, game_over, self.steps_counter # score is less with each step, hence 30 / steps_counter
    
    
    
    def step(self, action):
        # check if exited
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        # move

        self._move(action) # update the position   
        # reset step reward
        reward = 0
        # add reward for collecting _target_locations
        if self.check_target_location():
            reward = 1
            
        score, terminated, _ = self.game_over_check()
        reward += score
        
        # print(self.matrix)
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        # 6. return game over and score
        return observation, reward, terminated, False, info




    def _get_obs(self):
        # print({"targets_relative_line": self.check_for_target_line(), "targets_relative_general": self.check_for_target_general()})
        return {
            "agent": self._agent_location,
            "target": self.matrix.flatten(),
            "targets_relative_line": self.check_for_target_line(),
            "targets_relative_general": self.check_for_target_general()
            }

    def _get_info(self):
        return {
            "agent": self._agent_location,
            "target": self._target_locations
            }



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

        # First we draw the target
        for _target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * _target_location,
                    (pix_square_size, pix_square_size),
                ),
            )
            
        # Now we draw the agent
        
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i,j] > 0.5:
                    pygame.draw.rect(
                        canvas,
                        (100, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

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
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
