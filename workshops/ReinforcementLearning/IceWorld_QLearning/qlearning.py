import random

from ice_env import *
import torch

EPISODES = 1000
EPSILON = 0.15 #Exploration vs Exploitation
GAMMA = 0.9
LEARNING_RATE = 0.1


def main():
    env = Ice()
    average_cumulative_reward = 0.0

    # Q-table, for each env state have action space
    # (current example: 4x4 states, 4 actions per state)
    qtable = torch.zeros(env.env_space() + env.action_space(), dtype=torch.float)

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        
        # Loop over time-steps
        while not terminate:
            # 0 Currently you are in "state S (state)"
            # 1 Calculate action to be taken from state S. Use 'e-rand off-policy'
                # 1.1 Compute what the greedy action for the current state is
            qvalues = qtable[state]
            greedy_action = torch.argmax(qvalues)
                # 1.2 Sometimes, the agent takes a random action, to explore the environment
            if random.random() < EPSILON:
                action = random.randrange(4)
            else:
                action = greedy_action

            # 2 Perform the action
            next_state, reward, terminate = env.step(action)

            # 3 Update the q-table
            td_error = reward + GAMMA * max(qtable[next_state]) - qtable[state][action]
            qtable[state][action] += LEARNING_RATE * td_error

            # 4 Update cumulative reward
            cumulative_reward += reward

            # 5 Make current state next state
            state = next_state

        print(i, cumulative_reward)

    env.print_qtable_stats(qtable)

if __name__ == '__main__':
    main()