import gym
from pprint import pprint
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle


# Solve the enviroment using random chance, base method to build other methods on
class RandomMethod:
    def __init__(self, environment, use_learning_curve=True):
        self.use_learning_curve = use_learning_curve

        self.location_bins = 12
        self.velocity_bins = 12

        self.env = gym.make(environment)
        self.env._max_episode_steps = 200  # How long the simulation runs at max, should only be changed for testing

        # How long each episode took
        self.convergence_graph = []

    @staticmethod
    # Lower the chance to take a random action over time
    def select_exploration_chance(x):
        return max(.001, min(1.0, 1.0 - math.log10((x+1)/25)))

    @staticmethod
    # Lower the rate of change of the model over time
    def select_learning_rate(x):
        return max(.1, min(1.0, 1.0 - math.log10((x+1)/25)))

    # Convert the continuous data to discrete
    def bin_data(self, state_in):
        # Scale the position
        max_location = self.env.observation_space.high[0] + abs(self.env.observation_space.low[0])
        location_scaled = \
            math.floor((state_in[0] + abs(self.env.observation_space.low[0])) / max_location * self.location_bins)

        # Limit max and min values, else scale the velocity
        max_velocity = self.env.observation_space.high[1]
        if state_in[1] > max_velocity:
            velocity_scaled = 11
        elif state_in[1] < -max_velocity:
            velocity_scaled = 0
        else:
            velocity_scaled = math.floor((state_in[1] + max_velocity) / (max_velocity * 2) * self.velocity_bins)
        return location_scaled - 1, velocity_scaled - 1

    # Select an action to perform given a state and if the model is training
    def select_action(self, state):
        return self.env.action_space.sample()

    # load a pretrained model
    def load_model(self):
        # method.q_table = pickle.load(open("Best_Method.p", "rb"))
        # pickle.dump(method.q_table, open("Best_Method.p", "wb"))
        pass

    # Train the model to solve the problem
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
                # Do something random
                action = self.select_action(state)

                # run the simulation
                state, reward, done, info = self.env.step(action)

                # convert the continueous state data to discrete data
                location, velocity = self.bin_data(state)

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
            if timesteps < 130:
                streak += 1
                if streak > 2:
                    print("Found Streak at Episode: " + str(attempt))
                    break
            else:
                streak = 0

    # Evaluate the performance of the model
    def evaluate(self):
        attempts, total_timesteps = 100, 0
        num_optimal, min_timesteps = 0, 1000
        for attempt in range(attempts):
            timesteps = 0
            done = False

            # Run the simulation
            state = self.env.reset()
            while not done:
                angle, velocity = self.bin_data(state)
                action = self.select_action(state)
                state, reward, done, info = self.env.step(action)
                timesteps += 1

            # Log the performance for this run
            total_timesteps += timesteps
            if timesteps < min_timesteps:
                min_timesteps = timesteps
            if timesteps < 100:
                num_optimal += 1
            self.convergence_graph.append(timesteps)

        # pprint(self.q_table)
        print("Average timesteps per episode: " + str(total_timesteps / attempts))
        print("Minimum timesteps: " + str(min_timesteps))
        print("Optimal runs: " + str(num_optimal))
        return total_timesteps / attempts, min_timesteps, num_optimal

    # Run the program with the gui to show the trained model
    def display(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            action = self.select_action(state)
            state, reward, done, info = self.env.step(action)
        self.env.close()

    # plots the time each run took.
    def plot(self):
        plt.plot(self.convergence_graph)
        plt.ylabel("Timesteps Stable")
        plt.xlabel("Episode")
        plt.show()
