from q_learning import QLearning
from SARSA import SARSALearning
from method import RandomMethod

if __name__ == "__main__":
    # Initalize a method
    method = SARSALearning("MountainCar-v0")
    #method = QLearning("MountainCar-v0")
    # method = RandomMethod("MountainCar-v0")

    # Build the model
    load = False
    if load:
        method.load_model()
    else:
        method.train(1000)

    # Test the method
    method.evaluate()
    method.plot()
    method.display()


