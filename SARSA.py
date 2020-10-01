import gym
import random
import numpy as np
from method import RandomMethod

class SARSALearning(RandomMethod):
    def __init__(self, environment, use_learning_curve=True):
        super().__init__(environment, use_learning_curve)
        
        self.alpha = .5
        self.gamma = .9
        self.epsilon = .1
        
        self.env = gym.make(environment) #IS THERE A POINT IN THIS LINE?
        self.env._max_episode_steps = 500  # How long the simulation runs at max, should only be changed for testing

        # q_table holds the model that q learning uses to predict the best action
        self.q_table = np.zeros([self.location_bins, self.velocity_bins, self.env.action_space.n])
        
    def select_action(self, state, train=False):
        location, velocity = self.bin_data(state)
        
        if train:
            if random.uniform(0, 1) < self.epsilon:
                return self.env.action_space.sample()
            else:
                return np.argmax(self.q_table[location][velocity])
        else:
            return np.argmax(self.q_table[location][velocity])
            
    def train(self, max_attempts=1000):
        
        for attempt in range(max_attempts):
            if self.use_learning_curve:
                self.alpha=self.select_learning_rate(attempt)
                self.epsilon=self.select_exploration_chance(attempt)
                
            timeSteps=0
            done=False
            state=self.env.reset()
            location, velocity = self.bin_data(state)
            nextAction = self.select_action(state, True)
            while not done:
                startLocation = location
                startVelocity = velocity
                action = nextAction
                
                state, reward, done, info = self.env.step(action)
                location, velocity = self.bin_data(state)
                nextAction = self.select_action(state,True)
                
                originalVal = self.q_table[startLocation][startVelocity][action]
                self.q_table[startLocation][startVelocity][action] = originalVal + \
                    self.alpha * (reward + self.gamma * self.q_table[location][velocity][nextAction] - originalVal) 
                
                timeSteps+=1
                
                        # Print progress bar and then add data to graph
            if attempt % (max_attempts / 10) == 0:
                print("Training " + str(attempt / max_attempts * 100) + "% Complete.")
                pass
            self.convergence_graph.append(timeSteps)
