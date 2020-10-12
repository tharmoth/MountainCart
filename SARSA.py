import numpy as np
from q_learning import QLearning


class SARSALearning(QLearning):
    """
    Attempts to solve the mountain cart problem using SARSA.
    """
    def __init__(self, environment, print_progress=True):
        super().__init__(environment, print_progress)
        
        self.alpha = .9
        self.gamma = 1
        self.epsilon = .01

        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])

    def _model(self):
        # initializes all needed variables to train on a single episode
        state = self.env.reset()
        location, velocity = self._bin_data(state)
        next_action = self._select_action(state, True)
        done = False
        self.time_steps = 0
        # loops until the episode has completed
        while not done:
            start_location = location
            start_velocity = velocity
            action = next_action

            state, reward, done, info = self.env.step(action)
            location, velocity = self._bin_data(state)
            next_action = self._select_action(state, True)

            # SARSA updates: q(s,a) = q(s,a) + alpha*(reward + gamma*q(s+1, a+1) - q(s,a))
            original_val = self.q_table[start_location][start_velocity][action]
            self.q_table[start_location][start_velocity][action] = \
                original_val + self.alpha * \
                (reward + self.gamma * self.q_table[location][velocity][next_action] - original_val)

            self.time_steps += 1
