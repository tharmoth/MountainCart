import gym
import random
import numpy as np
from method import RandomMethod


class SARSALearning(RandomMethod):
    def __init__(self, environment, use_learning_curve=True):
        super().__init__(environment, use_learning_curve)
        
        self.alpha = .9
        self.gamma = 1
        self.epsilon = .01

        self.env._max_episode_steps = 200  # How long the simulation runs at max, should only be changed for testing

        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])
        
    # sets the epsilon greedy policy
    def select_action(self, state, train=False):
        location, velocity = self.bin_data(state)
        
        if train:
            if random.uniform(0, 1) < self.epsilon:
                return self.env.action_space.sample()
            else:
                return np.argmax(self.q_table[location][velocity])
        else:
            return np.argmax(self.q_table[location][velocity])
            
    def train(self, max_attempts=1000):
        streak = 0
        
        for attempt in range(max_attempts):
            if self.use_learning_curve:
                self.alpha = self.select_learning_rate(attempt)
                self.epsilon = self.select_exploration_chance(attempt)
            
            # initializes all needed variables to train on a single episode
            time_steps = 0
            done = False
            state = self.env.reset()
            location, velocity = self.bin_data(state)
            next_action = self.select_action(state, True)
            
            # loops until the episode has completed
            while not done:
                start_location = location
                start_velocity = velocity
                action = next_action
                
                state, reward, done, info = self.env.step(action)
                location, velocity = self.bin_data(state)
                next_action = self.select_action(state, True)
                
                # SARSA updates: q(s,a) = q(s,a) + alpha*(reward + gamma*q(s+1, a+1) - q(s,a))
                original_val = self.q_table[start_location][start_velocity][action]
                self.q_table[start_location][start_velocity][action] = original_val + \
                    self.alpha * (reward + self.gamma * self.q_table[location][velocity][next_action] - original_val)
                
                time_steps += 1
                
            # Print progress bar and then add data to graph
            if attempt % (max_attempts / 10) == 0:
                print("Training " + str(attempt / max_attempts * 100) + "% Complete.")
                pass
            self.convergence_graph.append(time_steps)
            
            if time_steps < 100:
                streak += 1
                if streak > 5:
                    print("Found Streak at Episode: " + str(attempt))
                    break
            else:
                streak = 0
