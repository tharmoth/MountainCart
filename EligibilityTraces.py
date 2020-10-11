import gym
import random
import numpy as np
from q_learning import QLearning
from pprint import pprint


class EligibilityTraces(QLearning):
    """
    Attempts to solve the mountain cart problem using SARSA.
    """
    def __init__(self, environment, print_progress=True):
        super().__init__(environment, print_progress)
        
        self.alpha = .9
        self.gamma = 1
        self.epsilon = .01
        self.rl_lambda = .5  # lambda

        # q_table holds the model that q learning uses to predict the best action, stores the state action values
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])
            
    def _model(self):
        # initializes all needed variables to train on a single episode
        state = self.env.reset()
        location, velocity = self._bin_data(state)
        next_action = self._select_action(state, True)
        done = False
        # stores the epsilon values
        epsilon_values = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])
        self.time_steps = 0
        # loops until the episode has completed
        while not done:
            start_location = location
            start_velocity = velocity
            action = next_action

            for e in epsilon_values:
                e = self.gamma * self.rl_lambda * e
            epsilon_values[start_location][start_velocity][action] = 1
            
            state, reward, done, info = self.env.step(action)
            location, velocity = self._bin_data(state)
            next_action = self._select_action(state, True)

            # SARSA updates: q(s,a) = q(s,a) + alpha*(reward + gamma*q(s+1, a+1) - q(s,a))
            s = reward + self.gamma * self.q_table[location][velocity][next_action] - \
                self.q_table[start_location][start_velocity][action]
            for locIndex, loc in enumerate(self.q_table):
                for velIndex, vel in enumerate(loc):
                    for actIndex, act in enumerate(vel):
                        self.q_table[locIndex][velIndex][actIndex] += \
                            self.alpha * s * epsilon_values[locIndex][velIndex][actIndex]

            self.time_steps += 1
