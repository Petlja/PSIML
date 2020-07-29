import random
import torch
import time
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from cartpole_env import CartPoleEnv
from dqn import DQN
from collections import deque

torch.set_default_dtype(torch.double)
random.seed(0)

#Hyper params
EPISODES = 500
STEPS = 300

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.985

#Discount rate
GAMMA = 0.95

#Switch nets frequency
SWITCH_FREQ = 10
#Update frequency
UPDATE_FREQ = 16

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.target = DQN(state_size, action_size)
        self.current = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.memory = deque(maxlen = 2000)
        self.batch_size =  64

        learning_rate = 0.0025
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.current(x)

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def remember(self, state, action, reward, next_state):
        # Remember (Q,S,A,R,S')
        self.memory.append((state, action, reward, next_state))

    def backward(self):
        # Experience replay
        # 1. Create mini batch (size: self.batch_size) for training
        # 2. Update the current net -> use target net to evalute target
            # Tip: For best performance you can use torch gradient accumulation
        for state, action, reward, next_state in random.sample(self.memory, self.batch_size):
            state_action_q_values = self.current.forward(torch.from_numpy(state))
            with torch.no_grad():
                next_state_action_q_values = self.target.forward(torch.from_numpy(next_state))

                state_action_q_values_target = torch.tensor(state_action_q_values)
                state_action_q_values_target[action] = reward + GAMMA * torch.max(next_state_action_q_values)

            loss = self.loss_fn(state_action_q_values, state_action_q_values_target)
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()


        
def main():
    env = CartPoleEnv()
    agent = DQNAgent(env.state_size(), env.action_size())

    epsilon = EPSILON_START
    results = []
    start = time.time()
    random.seed(0)

    for episode in range(EPISODES):
        #Start game/episode
        state = env.reset()

        if (episode > SWITCH_FREQ and episode % SWITCH_FREQ == 0):
            agent.update_target_model()

        #Loop inside one game episode
        for t in range(STEPS):
            # Display the game. Comment bellow line in order to get faster training.
            #env.render()

            state_action_q_values = agent.forward(torch.from_numpy(state))
            if random.random() <= epsilon:
                action = random.randrange(env.action_size())
            else:
                action = torch.argmax(state_action_q_values).item()
 
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state)

            if done or (t == STEPS - 1):
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, t, epsilon))
                results.append(t)
                break
            
            if episode > 10 and (episode + t) % UPDATE_FREQ == 0:
                agent.backward()

            state = next_state

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