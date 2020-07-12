import random
import numpy as np
import torch
import torch.nn as nn
from cartpole_env import CartPoleEnv
from dqn import DQN

torch.set_default_dtype(torch.double)

#Hyper params
EPISODES = 1000

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

#Discount rate
GAMMA = 0.95

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        learning_rate = 0.001
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

    for episode in range(EPISODES):
        #Start game/episode
        state = env.reset()

        #Loop inside one game episode for 500 steps
        for time in range(500):
            # Display the game. Comment bellow line in order to get faster training.
            #env.render()

            #0. Currently you are in "state S (state)"
            #1.1 Determine action q values from state S.
            #1.2 Calculate action to be taken from state S. Use 'e-rand off-policy'
            #1.3 Play/perform the action in the environment
                # Move to "next state S' (next_state), get reward, and flag for is game over (is new state terminal)
  
            state_action_q_values = agent.forward(torch.from_numpy(state))
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_size())
            else:
                action = torch.argmax(state_action_q_values).item()
 
            next_state, reward, done = env.step(action)

            #2.1 From state S' peek into the future - Determine action q values from state S'
            #2.2 Using the SARSA-MAX formula update the net. 
                # Suggestion: You can start with simplified formula: Q(S,A) <- R + gamma * max(Q(S',A'))
                # Hint1: Don't forget that you should only perform update for taken action (#1.2)
                # Hint2: Don't forget that the target is no_grad constant.
            with torch.no_grad():
                next_state_action_q_values = agent.forward(torch.from_numpy(next_state))

                state_action_q_values_target = torch.tensor(state_action_q_values)
                state_action_q_values_target[action] = reward + GAMMA * torch.max(next_state_action_q_values)

            agent.backward(state_action_q_values, state_action_q_values_target) #epochs, verbose

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, time, epsilon))
                break

            #3.1 Current state is now next_state
            #3.2 Apply epsilon decay
            state = next_state
            if epsilon > EPSILON_END:
                epsilon *= EPSILON_DECAY
            

if __name__ == '__main__':
    main()