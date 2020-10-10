import gym
import random
import numpy as np
import math
from method import RandomMethod


class QLearning(RandomMethod):
    """
    Attempts to solve the mountain cart problem using Q-Learning.

    Based on the q-learning method described in
    Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8.3-4 (1992): 279-292.
    """
    def __init__(self, environment):
        super().__init__(environment)

        self.location_bins = 12
        self.velocity_bins = 12

        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])

    # Reset's the model for multiple runs
    def reset_model(self):
        self.convergence_graph = []
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])

    # Convert the continuous data to discrete
    def _bin_data(self, state_in):
        # Scale the position
        max_location = self.env.observation_space.high[0] + abs(self.env.observation_space.low[0])
        location_scaled = \
            math.floor((state_in[0] + abs(self.env.observation_space.low[0])) / max_location * self.location_bins)

        # Limit max and min values, else scale the velocity
        max_velocity = self.env.observation_space.high[1]
        velocity_scaled = math.floor((state_in[1] + max_velocity) / (max_velocity * 2) * self.velocity_bins)
        return location_scaled - 1, velocity_scaled - 1

    # Sets the epsilon greedy policy
    def _select_action(self, state, train=False):
        location, velocity = self._bin_data(state)

        if train and random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[location][velocity])

    def _model(self):
        # initialize the model
        state = self.env.reset()
        location, velocity = self._bin_data(state)  # Position of Cart, Velocity of Cart
        done = False
        while not done:
            # either do something random or do the models best predicted action
            action = self._select_action(state, train=True)

            # save the old state
            location_old, velocity_old = location, velocity

            # run the simulation
            state, reward, done, info = self.env.step(action)

            # convert the continuous state data to discrete data
            location, velocity = self._bin_data(state)

            # Q(S,A) = Q(S,A) + a * [R + gamma * max_a(Q(S',A)) - Q(S,A)]
            # update the q learning model
            next_max = np.max(self.q_table[location][velocity])
            old_value = self.q_table[location_old][velocity_old][action]
            self.q_table[location_old][velocity_old][action] \
                += self.alpha * (reward + self.gamma * next_max - old_value)