import gym

import numpy as np
import math
import matplotlib.pyplot as plt


class RandomMethod:
    """
    Base method to contain shared classes between other methods. Sets up shared methods to solve and evaluate the
    mountain cart problem.
    """
    def __init__(self, environment):
        # Hyperparameters
        self.alpha = .9  # learning rate, how fast the model is trained.
        self.gamma = 1   # discount factor, immediate results or delayed results.
        self.epsilon = .01  # exploration chance, percent chance to take a random action.

        self.env = gym.make(environment)
        self._max_episode_steps = 500  # How long the simulation runs at max

        # How long each episode took
        self.convergence_graph = []

    # Train the model to solve the problem, attempts are episodes, time steps are epochs
    def train(self, max_attempts=1000):
        print("Training " + type(self).__name__)

        for attempt in range(max_attempts):
            # Lower the exploration rate and the learning rate over time.
            self.alpha = self._select_alpha(attempt)
            self.epsilon = self._select_gamma(attempt)

            # Run the training model
            self._model()

            # Print progress bar and then add data to graph
            self._print_progress(attempt, max_attempts)

            # Run attempt without random to evaluate progress
            self._evaluate_convergence(attempt)

    # Evaluate the performance of the model
    def evaluate(self):
        attempts, total_time_steps = 100, 0
        num_optimal, min_time_steps = 0, 1000
        for attempt in range(attempts):
            time_steps = 0
            done = False

            # Run the simulation
            state = self.env.reset()
            while not done:
                action = self._select_action(state)
                state, reward, done, info = self.env.step(action)
                time_steps += 1

            # Log the performance for this run
            total_time_steps += time_steps
            if time_steps < min_time_steps:
                min_time_steps = time_steps
            if time_steps < 100:
                num_optimal += 1

        # pprint(self.q_table)
        print("Average time steps per episode: " + str(total_time_steps / attempts))
        print("Minimum time steps: " + str(min_time_steps))
        print("Optimal runs: " + str(num_optimal))
        return total_time_steps / attempts, min_time_steps, num_optimal

    # Run the program once with the graphics to display the trained model
    def display(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            action = self._select_action(state)
            state, reward, done, info = self.env.step(action)
        self.env.close()

    # plots the time each run took.
    def plot(self):
        plt.plot(self.convergence_graph)
        plt.ylabel("Time steps to solve")
        plt.xlabel("Episode")
        plt.show()

        plt.hist(self.convergence_graph, 20)
        plt.show()

    # Reset's the model for multiple runs
    def reset_model(self):
        self.convergence_graph = []

    # Trains the reinforcement learning method
    def _model(self):
        # initialize the model
        state = self.env.reset()
        done = False
        while not done:
            # Do something random
            action = self._select_action(state)

            # run the simulation
            state, reward, done, info = self.env.step(action)

    @staticmethod
    # Lower the chance to take a random action over time
    def _select_gamma(x):
        return max(.05, min(1.0, 1.0 - math.log10((x+1)/25)))

    @staticmethod
    # Lower the rate of change of the model over time
    def _select_alpha(x):
        return max(.1, min(1.0, 1.0 - math.log10((x+1)/25)))

    @staticmethod
    # Prints how much progress has occurred based on the number of time steps and max time steps.
    def _print_progress(attempt, max_attempts):
        if attempt % (max_attempts / 4) == 0:
            print("Training " + str(attempt / max_attempts * 100) + "% Complete.")
            pass

    # Runs the model once greedily to evaluate the performance of the model over time.
    def _evaluate_convergence(self, attempt):
        if attempt % 10 == 0:
            state = self.env.reset()
            time_steps = 0
            done = False
            while not done:
                action = self._select_action(state)
                state, reward, done, info = self.env.step(action)
                time_steps += 1
            self.convergence_graph = np.append(self.convergence_graph, time_steps * np.ones(1))

    # Select an action to perform given a state and if the model is training
    def _select_action(self, state):
        return self.env.action_space.sample()
