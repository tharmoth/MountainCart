import random
import numpy as np
from method import RandomMethod


class FApprox(RandomMethod):
    """
    Attempts to solve the mountain cart problem using tile coding and SARSA.

    Based on the tile coding from
    Watkins, C.J.C.H. (1989), Learning from Delayed Rewards (PDF) (Ph.D. thesis), Cambridge University pp. 140-146
    """
    def __init__(self, environment, print_progress=True):
        super().__init__(environment, print_progress)

        self.tiles = 12
        self.subtiles = 12

        self.weights = np.zeros([self.tiles, self.subtiles, self.subtiles, self.env.action_space.n])
        self.active_tiles_prime = None  # Used so tileizer is only called once per time step

    # Reset's the model for multiple runs
    def reset_model(self):
        self.convergence_graph = []
        self.weights = np.zeros([self.tiles, self.subtiles, self.subtiles, self.env.action_space.n])
        self.active_tiles_prime = None  # Used so tileizer is only called once per time step

    # Determine which tiles the cart is in
    def _tileizer(self, state):
        x = state[0]
        y = state[1]

        max_width = self.env.observation_space.high[0]
        min_width = self.env.observation_space.low[0]

        max_height = self.env.observation_space.high[1]
        min_height = -self.env.observation_space.high[1]

        width = (max_width - min_width) / (self.subtiles - 1)
        height = (max_height - min_height) / (self.subtiles - 1)

        width_offset = width / self.tiles
        height_offset = height / self.tiles

        tile_indexes = []

        for tiles_num in range(self.tiles):
            x_index = -1
            y_index = -1
            for x_tile in range(self.subtiles):
                x_tile_next = x_tile + 1
                min_loc = min_width + (tiles_num * width_offset) + (x_tile * width)
                max_loc = min_width + (tiles_num * width_offset) + (x_tile_next * width)
                if min_loc < x < max_loc:
                    x_index = x_tile
                    break
            for y_tile in range(self.subtiles):
                y_tile_next = y_tile + 1
                min_loc = min_height + (tiles_num * height_offset) + (y_tile * height)
                max_loc = min_height + (tiles_num * height_offset) + (y_tile_next * height)
                if min_loc < y < max_loc:
                    y_index = y_tile
                    break

            tile_indexes.append([tiles_num, x_index, y_index])
        return tile_indexes

    # Select an action to perform given a state and if the model is training use epsilon greedy
    def _select_action(self, state, train=False):
        tile_indexes = self.active_tiles_prime
        if tile_indexes is None:
            tile_indexes = self._tileizer(state)

        if train and random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore action space
        else:
            # Select the action with the highest combined weight
            left, none, right = 0, 0, 0
            for tile in tile_indexes:
                left += self.weights[tile[0]][tile[1]][tile[2]][0]
                none += self.weights[tile[0]][tile[1]][tile[2]][1]
                right += self.weights[tile[0]][tile[1]][tile[2]][2]

            return np.argmax([left, none, right])  # Exploit learned values

    def _model(self):
        # initialize the model
        state = self.env.reset()
        self.active_tiles_prime = self._tileizer(state)
        action_prime = self._select_action(state, train=True)
        done = False
        self.time_steps = 0
        while not done:
            # save the old state
            active_tiles = self.active_tiles_prime
            action = action_prime

            # run the simulation
            state, reward, done, info = self.env.step(action_prime)

            # convert the continuous state data to discrete data
            self.active_tiles_prime = self._tileizer(state)

            # w = w + a * [R + gamma * Q(S',A', w) - Q(S,A, w)] * x(s)
            # update the weights for each tile the cart is in
            for i in range(self.tiles):
                tile = active_tiles[i]
                tile_prime = self.active_tiles_prime[i]
                q_hat_prime = self.weights[tile_prime[0]][tile_prime[1]][tile_prime[2]][action_prime]
                q_hat = self.weights[tile[0]][tile[1]][tile[2]][action]

                self.weights[tile[0]][tile[1]][tile[2]][action] += \
                    self.alpha * (reward + self.gamma * q_hat_prime - q_hat)

            # either do something random or do the models best predicted action epsilon-greedy
            action_prime = self._select_action(state, train=True)

            self.time_steps += 1
        self.active_tiles_prime = None
