import gym
from pprint import pprint
import random
import numpy as np
import math
import matplotlib.pyplot as plt


class RLearner:
    def __init__(self, environment, use_learning_curve=True):
        # Hyperparameters
        self.alpha = .5
        self.gamma = .999
        self.epsilon = 0.2
        self.angle_bins = 6
        self.velocity_bins = 12
        self.use_learning_curve = use_learning_curve

        self.env = gym.make(environment)
        self.q_table = np.zeros([self.angle_bins, self.velocity_bins, self.env.action_space.n])
        self.convergence_graph = []

    @staticmethod
    # change the exploration rate over time.
    def select_explore_rate(x):
        return max(.1, min(1.0, 1.0 - math.log10((x+1)/25)))

    # Change learning rate over time
    @staticmethod
    def select_learning_rate(x):
        return max(.1, min(1.0, 1.0 - math.log10((x+1)/25)))

    # Convert the continuous data to discrete
    def bin_data(self, state_in):
        max_angle = self.env.observation_space.high[2] / 2
        angle_scaled = math.floor((state_in[2] + max_angle) / (max_angle * 2) * self.angle_bins)

        max_velocity = .9
        if state_in[3] > max_velocity:
            velocity_scaled = 11
        elif state_in[3] < -max_velocity:
            velocity_scaled = 0
        else:
            velocity_scaled = math.floor((state_in[3] + max_velocity) / (max_velocity * 2) * self.velocity_bins)
        return angle_scaled - 1, velocity_scaled - 1

    def train(self):
        for i in range(1, 1000):
            if self.use_learning_curve:
                # learning rate and explore rate diminishes monotonically over time
                self.alpha = self.select_explore_rate(i)
                self.epsilon = self.select_learning_rate(i)

            state = self.env.reset()

            epochs, reward, = 0, 0
            done = False

            while not done:
                angle_old, velocity_old = self.bin_data(state)
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(self.q_table[angle_old][velocity_old])  # Exploit learned values

                next_state, reward, done, info = self.env.step(action)

                angle, velocity_new = self.bin_data(next_state)
                next_max = np.max(self.q_table[angle][velocity_new])

                old_value = self.q_table[angle_old][velocity_old][action]

                self.q_table[angle_old][velocity_old][action] += self.alpha * (reward + self.gamma * next_max - old_value)

                state = next_state
                epochs += 1

            self.convergence_graph.append(epochs)

        print("Training finished.")

    # evaluate the performance of the model
    def evaluate(self):
        total_epochs = 0
        episodes = 1000
        # convergence_graph = []
        for _ in range(episodes):
            state = self.env.reset()
            epochs, reward = 0, 0

            done = False

            while not done:
                angle, velocity = self.bin_data(state)
                action = np.argmax(self.q_table[angle][velocity])
                state, reward, done, info = self.env.step(action)

                epochs += 1
            self.convergence_graph.append(epochs)
            total_epochs += epochs

        #pprint(self.q_table)
        print("Results after " + str(episodes) + " episodes:")
        print("Average timesteps per episode: " + str(total_epochs / episodes))

    def display(self):
        observation = self.env.reset()

        for t in range(500):
            self.env.render()

            angle, velocity = self.bin_data(observation)
            action = np.argmax(self.q_table[angle][velocity])
            observation, reward, done, info = self.env.step(action)
            if done:
                break
        self.env.close()

    def plot(self):
        plt.plot(self.convergence_graph)
        plt.ylabel("Timesteps Stable")
        plt.xlabel("Episode")
        plt.show()


if __name__ == "__main__":
    method = RLearner("CartPole-v1")
    method.train()
    method.evaluate()
    method.plot()
    method.display()
