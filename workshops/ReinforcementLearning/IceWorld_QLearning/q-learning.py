import random

from ice import *
import numpy
import time

EPISODES = 25000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.1

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]
    
def main():
    env = Ice()
    average_cumulative_reward = 0.0

    # Q-table, 4x4 states, 4 actions per state
    qtable = [[0., 0., 0., 0.] for state in range(4*4)]

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        
        # TO DO
        # Loop over time-steps
        while not terminate:
            # Compute what the greedy action for the current state is
            qvalues = qtable[state]
            greedy_action = argmax(qvalues)

            # Sometimes, the agent takes a random action, to explore the environment
            if random.random() < EPSILON:
                action = random.randrange(4)
            else:
                action = greedy_action

            # Perform the action
            next_state, reward, terminate = env.step(action)

            # Update the Q-Table
            td_error = reward + GAMMA * max(qtable[next_state]) - qtable[state][action]
            qtable[state][action] += LEARNING_RATE * td_error

            # Update statistics
            cumulative_reward += reward
            state = next_state

        print(i, cumulative_reward)

    print(qtable)

if __name__ == '__main__':
    main()