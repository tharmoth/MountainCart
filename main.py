from q_learning import QLearning
from SARSA import SARSALearning
from eligibility_traces import EligibilityTraces
from function_approximation import FApprox
from mountain_cart import run_methods, self_iterate
import pickle

if __name__ == "__main__":
    # Initialize a method
    methods = [
               QLearning("MountainCar-v0", print_progress=False),
               SARSALearning("MountainCar-v0", print_progress=False),
               FApprox("MountainCar-v0", print_progress=False),
               EligibilityTraces("MountainCar-v0", print_progress=False)
               ]

    # Run the tests
    run_methods(methods)

    method = methods[0]
    method.q_table = pickle.load(open("Best_Method_" + str(type(method).__name__) + ".p", "rb"))
    method.evaluate()
    method.display()

    self_iterate(methods[0])

