import random
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from cartpole_env import CartPoleEnv
from dqn import DQN

torch.set_default_dtype(torch.double)
random.seed(0)

#Hyper params
EPISODES = 500
STEPS = 300

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.99

#Discount rate
GAMMA = 0.95

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        learning_rate = 0.0025
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

    def backward(self, y, y_target):
        loss = self.loss_fn(y, y_target)

        # Zero-out all the gradients
        self.optimizer.zero_grad()

        # Backward pass: compute gradient of the loss
        loss.backward()

        # Calling the step function on an Optimizer in order to apply the gradients (update)
        self.optimizer.step()

        
def main():
    env = CartPoleEnv()
    agent = DQNAgent(env.state_size(), env.action_size())

    epsilon = EPSILON_START
    results = []
    start = time.time()

    for episode in range(EPISODES):
        #Start game/episode
        state = env.reset()

        #Loop inside one game episode
        for t in range(STEPS):
            # Display the game. Comment bellow line in order to get faster training.
            env.render()

            #0. Currently you are in "state S (state)"
            #1.1 Determine action q values from state S.
            #1.2 Calculate action to be taken from state S. Use 'e-rand off-policy'
            #1.3 Play/perform the action in the environment
                # Move to "next state S' (next_state), get reward, and flag for is game over (is new state terminal)
  
            pass
            done = True #Update this flag correctly

            #2.1 From state S' peek into the future - Determine action q values from state S'
            #2.2 Using the SARSA-MAX formula update the net. 
                # Suggestion: You can start with formula: Q(S,A) <- R + gamma * max(Q(S',A'))
                # Hint1: Don't forget that you should only perform update for taken action (#1.2)
                # Hint2: Don't forget that the target is no_grad constant.
            pass

            if done or (t == STEPS - 1):
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, t, epsilon))
                results.append(t)
                break

            #3.1 Current state is now next_state
            pass

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

    end = time.time()
    print("TIME")
    print(end - start)
    print("STEPS")
    print(sum(results))
    plt.plot(results)
    plt.show()

if __name__ == '__main__':
    main()