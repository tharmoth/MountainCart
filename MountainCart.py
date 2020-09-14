import gym
from pprint import pprint
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle


class QLearning:
    def __init__(self, environment, use_learning_curve=True):
        # Hyperparameters
        self.alpha = .8
        self.gamma = 1
        self.epsilon = .1
        self.angle_bins = 12
        self.velocity_bins = 12
        self.use_learning_curve = use_learning_curve

        self.env = gym.make(environment)
        self.env._max_episode_steps = 200
        self.q_table = np.zeros([self.angle_bins, self.velocity_bins, self.env.action_space.n])
        self.convergence_graph = []

    @staticmethod
    # change the exploration rate over time.
    def select_explore_rate(x):
        return max(.001, min(1.0, 1.0 - math.log10((x+1)/25)))

    # Change learning rate over time
    @staticmethod
    def select_learning_rate(x):
        return max(.1, min(1.0, 1.0 - math.log10((x+1)/25)))

    # Convert the continuous data to discrete
    def bin_data(self, state_in):
        max_angle = self.env.observation_space.high[0] + abs(self.env.observation_space.low[0])
        angle_scaled = math.floor((state_in[0] + abs(self.env.observation_space.low[0])) / max_angle * self.angle_bins)

        max_velocity = self.env.observation_space.high[1]
        if state_in[1] > max_velocity:
            velocity_scaled = 11
        elif state_in[1] < -max_velocity:
            velocity_scaled = 0
        else:
            velocity_scaled = math.floor((state_in[1] + max_velocity) / (max_velocity * 2) * self.velocity_bins)
        return angle_scaled - 1, velocity_scaled - 1

    def train(self):
        streak = 0
        max_iterations = 10000
        for episode in range(max_iterations):
            if self.use_learning_curve:
                # learning rate and explore rate diminishes monotonically over time
                self.alpha = self.select_learning_rate(episode)
                self.epsilon = self.select_explore_rate(episode)

            state = self.env.reset()

            epochs, reward, = 0, 0
            done = False

            while not done:
                # self.env.render()
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

            if epochs < 130:
                streak += 1
            else:
                streak = 0

            if streak > 2:
                print("Found Streak at Episode: " + str(episode))
                break

            if epochs < 100:
                print("Optimal Detected")
                # break

            if episode % (max_iterations / 10) == 0:
                # print("Training " + str(i / max_iterations * 100) + "% Complete.")
                pass
            self.convergence_graph.append(epochs)

       # print("Training Complete.")

    # evaluate the performance of the model
    def evaluate(self):
        total_epochs = 0
        episodes = 100
        min_timesteps = 1000
        num_optimal = 0
        # convergence_graph = []
        for _ in range(episodes):
            state = self.env.reset()
            epochs, reward = 0, 0

            done = False

            while not done:
                self.env.render()
                # self.env.render()
                angle, velocity = self.bin_data(state)
                action = np.argmax(self.q_table[angle][velocity])
                state, reward, done, info = self.env.step(action)

                epochs += 1
            self.convergence_graph.append(epochs)
            total_epochs += epochs
            if epochs < min_timesteps:
                min_timesteps = epochs
            if epochs < 100:
                num_optimal += 1

        # pprint(self.q_table)
        print("Average timesteps per episode: " + str(total_epochs / episodes))
        print("Minimum timesteps: " + str(min_timesteps))
        print("Optimal runs: " + str(num_optimal))
        return total_epochs / episodes, min_timesteps, num_optimal

    def display(self):
        observation = self.env.reset()

        for t in range(1000):
            self.env.render()

            angle, velocity = self.bin_data(observation)
            action = np.argmax(self.q_table[angle][velocity])
            observation, reward, done, info = self.env.step(action)
            # if done:
            #     break
        self.env.close()

    def plot(self):
        plt.plot(self.convergence_graph)
        plt.ylabel("Timesteps Stable")
        plt.xlabel("Episode")
        plt.show()


class RandomMethod:
    def __init__(self, environment):
        self.env = gym.make(environment)

    def display(self):

        self.env.reset()
        while True:
            self.env.render()

            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
        self.env.close()


if __name__ == "__main__":
    method = QLearning("MountainCar-v0", True)
    # method.q_table = pickle.load(open("venv/Best_Method.p", "rb"))
    method.train()
    run_average, run_minimum, run_optimal = method.evaluate()
    # pickle.dump(method.q_table, open("Best_Method.p", "wb"))

    method.plot()
    method.display()

    # method = RandomMethod("MountainCar-v0")
    # method.display()


