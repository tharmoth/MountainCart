from q_learning import QLearning
from SARSA import SARSALearning
from eligibility_traces import EligibilityTraces
from function_approximation import FApprox
import matplotlib.pyplot as plt
import numpy as np
import pickle


def run_method(method):
    method = methods[0]
    method.train(1000)
    # Test the method
    method.evaluate()
    method.plot()


def run_methods(methods_in):
    for method in methods_in:
        episodes = 500
        runs = 1
        graph = np.zeros(int(episodes / 10))
        average_a, minimum_a, optimal_a = 0, 0, 0
        for run in range(1, runs + 1):
            print(type(method).__name__ + " run number " + str(run))
            method.reset_model()
            method.train(episodes)
            graph += method.convergence_graph / runs
            average, minimum, optimal = method.evaluate()
            average_a += average / runs
            minimum_a += minimum / runs
            optimal_a += optimal / runs

        plt.plot(graph, label=type(method).__name__)
        print("Average: " + str(average_a) +
              ", Minimum: " + str(minimum_a) +
              ", Optimal: " + str(optimal_a))

    plt.legend()
    plt.ylabel("Timesteps To Solve")
    plt.xlabel("Epoch")
    plt.show()


def self_iterate(method):
    optimal = 1000
    best_optimal = 0
    best_minimum = 1000
    best_average = 1000

    print("Running self iterate, This will run and save the best result until the user exits the program.")

    try:
        try:
            method.q_table = pickle.load(open("Best_Method_" + str(type(method).__name__) + ".p", "rb"))
            best_average, best_minimum, best_optimal = method.evaluate()
        except:
            pass
        while True:
            method.reset_model()
            try:
                method.q_table = pickle.load(open("Best_Method_" + str(type(method).__name__) + ".p", "rb"))
            except:
                pass
            method.train(500, break_on_trained=True)
            average, minimum, optimal = method.evaluate()

            if best_optimal == 0:
                if average < best_average:
                    best_average = average
                    pickle.dump(method.q_table, open("Best_Method_" + str(type(method).__name__) + ".p", "wb"))
                    print("New best found!")
            elif optimal > best_optimal:
                best_optimal = optimal
                best_average = average
                pickle.dump(method.q_table, open("Best_Method_" + str(type(method).__name__) + ".p", "wb"))
                print("New best found!")
            elif optimal == best_optimal and average < best_average:
                best_optimal = optimal
                best_average = average
                pickle.dump(method.q_table, open("Best_Method_" + str(type(method).__name__) + ".p", "wb"))
                print("New best found!")
    except KeyboardInterrupt:
        method.reset_model()
        method.q_table = pickle.load(open("Best_Method_" + str(type(method).__name__) + ".p", "rb"))
        method.evaluate()
        method.plot()
        method.display()


if __name__ == "__main__":
    # Initialize a method
    methods = [
               QLearning("MountainCar-v0", print_progress=False),
               SARSALearning("MountainCar-v0", print_progress=False),
               FApprox("MountainCar-v0", print_progress=False),
               EligibilityTraces("MountainCar-v0", print_progress=False)
               ]

    # run_method(methods[3])
    # self_iterate(methods[0])
    run_methods(methods)

    # method = methods[3]
    # method.q_table = pickle.load(open("Best_Method_" + str(type(method).__name__) + ".p", "rb"))
    # method.plot()
    # method.evaluate()
    # method.display()
