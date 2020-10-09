import gym
import random
import numpy as np
import pickle
import math
from method import RandomMethod


# Solve the environment using function approximation and SARSA
class FApprox(RandomMethod):
    def __init__(self, environment, use_learning_curve=True):
        super().__init__(environment, use_learning_curve)

        # Hyperparameters
        self.learning_rate = .8  # Alpha, how fast the model is trained.
        self.discount_factor = 1  # Gamma, immediate results or delayed results.
        self.exploration_chance = .1  # Epsilon, percent chance to take a random action.

        self.env = gym.make(environment)
        self.env._max_episode_steps = 200  # How long the simulation runs at max, should only be changed for testing
        self.tiles = 8
        self.subtiles = 8
        self.weights = np.zeros([self.tiles, self.subtiles, self.subtiles, self.env.action_space.n])
        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])

    # Convert the continuous data to discrete
    def bin_data(self, state_in):
        # Scale the position
        max_location = self.env.observation_space.high[0] + abs(self.env.observation_space.low[0])
        location_scaled = \
            math.floor((state_in[0] + abs(self.env.observation_space.low[0])) / max_location * self.location_bins)

        # Limit max and min values, else scale the velocity
        max_velocity = self.env.observation_space.high[1]
        velocity_scaled = math.floor((state_in[1] + max_velocity) / (max_velocity * 2) * self.velocity_bins)
        return location_scaled - 1, velocity_scaled - 1

    # Determine which tiles the cart is in
    def tileizer(self, state):
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
            x_index = 0
            y_index = 0
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

    # load a pretrained model
    def load_model(self):
        self.q_table = pickle.load(open("Best_Method.p", "rb"))

    # Select an action to perform given a state and if the model is training
    def select_action(self, state, train=False):
        tile_indexes = self.tileizer(state)

        if train and random.uniform(0, 1) < self.exploration_chance:
            return self.env.action_space.sample()  # Explore action space
        else:
            left, none, right = 0, 0, 0
            for tile in tile_indexes:
                left += self.weights[tile[0]][tile[1]][tile[2]][0]
                none += self.weights[tile[0]][tile[1]][tile[2]][1]
                right += self.weights[tile[0]][tile[1]][tile[2]][2]

            return np.argmax([left, none, right])  # Exploit learned values

    # Train the model to solve the problem, attempts are episodes, timesteps are epochs
    def train(self, max_attempts=1000):
        streak = 0
        for attempt in range(max_attempts):

            # Lower the exploration rate and the learning rate over time.
            if self.use_learning_curve:
                self.learning_rate = self.select_learning_rate(attempt)
                self.exploration_chance = self.select_exploration_chance(attempt)

            timesteps = 0
            done = False

            # initialize the model
            state = self.env.reset()
            tile_indexes = self.tileizer(state)
            action = self.select_action(state, train=True)

            while not done:
                # save the old state
                tile_indexes_old = tile_indexes
                action_old = action

                # run the simulation
                state, reward, done, info = self.env.step(action)

                # convert the continuous state data to discrete data
                tile_indexes = self.tileizer(state)

                for i in range(self.tiles):
                    t_old = tile_indexes_old[i]
                    t_new = tile_indexes[i]
                    qhat_prime = self.weights[t_new[0]][t_new[1]][t_new[2]][action]
                    qhat = self.weights[t_old[0]][t_old[1]][t_old[2]][action_old]

                    self.weights[t_old[0]][t_old[1]][t_old[2]][action_old] += \
                        self.learning_rate * (reward + self.discount_factor * qhat_prime - qhat)

                # either do something random or do the models best predicted action
                action = self.select_action(state, train=True)

                # number of timesteps taken to solve
                timesteps += 1

            # Print progress bar and then add data to graph
            if attempt % (max_attempts / 10) == 0:
                print("Training " + str(attempt / max_attempts * 100) + "% Complete.")
                pass
            self.convergence_graph.append(timesteps)

            # The rest of the code are arbitrary conditions that signal the model is trained
            # I was playing around with them and these conditions seem to yield good results most of the time
            # Feel free to play around with this as much as you'd like
            if timesteps < 180:
                streak += 1
                if streak > 5:
                    print("Found Streak at Episode: " + str(attempt))
                    break
            else:
                streak = 0
