from q_learning import QLearning
from SARSA import SARSALearning
from ElgibilityTraces import ElgibilityTraces
from function_approximation import FApprox
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Initialize a method
    methods = [QLearning("MountainCar-v0"),
               SARSALearning("MountainCar-v0"),
               FApprox("MountainCar-v0"),
               ElgibilityTraces("MountainCar-v0")]
    method = methods[3]
    method.train(1000)

    # Test the method
    method.evaluate()
    method.plot()

    #for method in methods:
    #    graph = np.zeros(method.env._max_episode_steps / 10)
    ##    runs = 30
     #   for run in range(1, runs + 1):
     #       method.reset_model()
    #        method.train(1000)
     
     #       graph += method.convergence_graph / runs

      #  plt.plot(graph, label=type(method).__name__)

    #plt.legend()
    #plt.ylabel("Timesteps To Solve")
    #plt.xlabel("Episode")
    #plt.show()
