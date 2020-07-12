import random
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from cartpole_env import CartPoleEnv
from dqn import DQN
from collections import deque

torch.set_default_dtype(torch.double)

#Hyper params
EPISODES = 1000

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999

#Discount rate
GAMMA = 0.95

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.target = DQN(state_size, action_size)
        self.current = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.memory = deque(maxlen = 2000)
        self.batch_size = 64

        learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.current(x)

    def update_target_model(self):
        # Target becomes current
        pass

    def remember(self, state, action, reward, next_state):
        # Remember (Q,S,A,R,S')
        pass

    def backward(self):
        # Expirience replay
        # 1. Create mini batch (size: self.batch_size) for training
        # 2. Update the current net -> use target net to evalute target
            # Tip: For best performance you can use torch gradient accumulation
        pass


        
def main():
    env = CartPoleEnv()
    agent = DQNAgent(env.state_size(), env.action_size())

    epsilon = EPSILON_START

    for episode in range(EPISODES):
        #Start game/episode
        state = env.reset()

        if (episode > 10 and episode % 10 == 0):
            agent.update_target_model()

        #Loop inside one game episode for 500 steps
        for time in range(500):
            # Display the game. Comment bellow line in order to get faster training.
            #env.render()

            state_action_q_values = agent.forward(torch.from_numpy(state))
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_size())
            else:
                action = torch.argmax(state_action_q_values).item()
 
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state)

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, time, epsilon))
                break
            
            if episode > 10 and (episode + time) % 16 == 0:
                agent.backward()

            state = next_state
            if epsilon > EPSILON_END:
                epsilon *= EPSILON_DECAY
            

if __name__ == '__main__':
    main()