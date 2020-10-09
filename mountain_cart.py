from q_learning import QLearning
from SARSA import SARSALearning
from function_approximation import FApprox
from method import RandomMethod

if __name__ == "__main__":
    # Initialize a method
    # method = QLearning("MountainCar-v0")
    # method = SARSALearning("MountainCar-v0")
    method = FApprox("MountainCar-v0")
    # method = RandomMethod("MountainCar-v0")

    # Build the model
    load = False
    if load:
        method.load_model()
    else:
        method.train()

    # Test the method
    method.evaluate()
    method.plot()
    method.display()


