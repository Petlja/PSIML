import random

from ice_env import *
import torch
import time

EPISODES = 2000
EPSILON = 0.15 #Exploration vs Exploatation
GAMMA = 0.9
LEARNING_RATE = 0.1

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]
    
def main():
    env = Ice()
    average_cumulative_reward = 0.0

    # Q-table, for each env state have action space
    # (current example: 4x4 states, 4 actions per state)
    qtable = torch.zeros(env.env_space() + env.action_space(),dtype=torch.float)

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        
        # Loop over time-steps
        while terminate:
            # 0 Currently you are in "state S (state)"
            # 1 Calculate action to be taken from state S. Use 'e-rand off-policy'
                # 1.1 Compute what the greedy action for the current state is
            pass

                # 1.2 Sometimes, the agent takes a random action, to explore the environment
            pass

            # 2 Perform the action
            pass

            # 3 Update the q-table
            pass

            # 4 Update cumulative reward
            pass

            # 5 Make current state next state
            pass

        print(i, cumulative_reward)

    env.print_qtable_stats(qtable)

if __name__ == '__main__':
    main()