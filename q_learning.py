import gym
import random
import numpy as np
import pickle
from method import RandomMethod


# Solve the enviroment using q learning
class QLearning(RandomMethod):
    def __init__(self, environment, use_learning_curve=True):
        super().__init__(environment, use_learning_curve)

        # Hyperparameters
        self.learning_rate = .8  # Alpha, how fast the model is trained.
        self.discount_factor = 1  # Gamma, immident results or delayed results.
        self.exploration_chance = .1  # Epsilon, percent chance to take a random action.

        self.env = gym.make(environment)
        self.env._max_episode_steps = 500  # How long the simulation runs at max, should only be changed for testing

        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])

    # load a pretrained model
    def load_model(self):
        self.q_table = pickle.load(open("Best_Method.p", "rb"))

    # Select an action to perform given a state and if the model is training
    def select_action(self, state, train=False):
        location, velocity = self.bin_data(state)

        if train:
            # either do something random or do the models best predicted action
            if random.uniform(0, 1) < self.exploration_chance:
                action = self.env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(self.q_table[location][velocity])  # Exploit learned values
        else:
            action = np.argmax(self.q_table[location][velocity])
        return action

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

            # initalize the model
            state = self.env.reset()
            location, velocity = self.bin_data(state)  # Position of Cart, Velocity of Cart

            while not done:
                # either do something random or do the models best predicted action
                action = self.select_action(state, train=True)

                # save the old state
                location_old, velocity_old = location, velocity

                # run the simulation
                state, reward, done, info = self.env.step(action)

                # convert the continueous state data to discrete data
                location, velocity = self.bin_data(state)

                # update the q learning model
                next_max = np.max(self.q_table[location][velocity])
                old_value = self.q_table[location_old][velocity_old][action]
                self.q_table[location_old][velocity_old][action] += \
                    self.learning_rate * (reward + self.discount_factor * next_max - old_value)

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
