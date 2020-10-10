import gym
import random
import numpy as np
from q_learning import QLearning
from pprint import pprint

class ElgibilityTraces(QLearning):
    """
    Attempts to solve the mountain cart problem using SARSA.
    """
    def __init__(self, environment):
        super().__init__(environment)
        
        self.alpha = .9
        self.gamma = 1
        self.epsilon = .01
        self.l = .5 #lambda

        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n]) #stores the state action vals
            
    def _model(self):
        # initializes all needed variables to train on a single episode
        state = self.env.reset()
        location, velocity = self._bin_data(state)
        next_action = self._select_action(state, True)
        done = False
        epsilonVals = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n]) # stores the epsilon vals
        # loops until the episode has completed
        while not done:
            start_location = location
            start_velocity = velocity
            action = next_action

            for e in epsilonVals:
                e = self.gamma * self.l * e
            epsilonVals[start_location][start_velocity][action] = 1
            
            state, reward, done, info = self.env.step(action)
            location, velocity = self._bin_data(state)
            next_action = self._select_action(state, True)

            # SARSA updates: q(s,a) = q(s,a) + alpha*(reward + gamma*q(s+1, a+1) - q(s,a))
            s = reward + self.gamma * self.q_table[location][velocity][next_action] - self.q_table[start_location][start_velocity][action]
            for locIndex, loc in enumerate(self.q_table):
                for velIndex, vel in enumerate(loc):
                    for actIndex, act in enumerate(vel):
                        self.q_table[locIndex][velIndex][actIndex] += self.alpha * s * epsilonVals[locIndex][velIndex][actIndex]
