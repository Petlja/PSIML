import random
import gym
from gym import wrappers

EPISODES = 100000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.1
DISCRETE_STEPS = 20     # 10 discretization steps per state variable


def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]
    
def make_state(observation):
    """ Map a 4-dimensional state to a state index
    """
    low = [-4.8, -10., -0.41, -10.]
    high = [4.8, 10., 0.41, 10.]
    state = 0

    for i in range(4):
        # State variable, projected to the [0, 1] range
        state_variable = (observation[i] - low[i]) / (high[i] - low[i])

        # Discretize. A variable having a value of 0.53 will lead to the integer 5,
        # for instance.
        state_discrete = int(state_variable * DISCRETE_STEPS)
        state_discrete = max(0, state_discrete)
        state_discrete = min(DISCRETE_STEPS-1, state_discrete)

        state *= DISCRETE_STEPS
        state += state_discrete

    return state
    
def main():
    average_cumulative_reward = 0.0

    # Create the Gym environment (CartPole)
    env = gym.make('CartPole-v1')
    env = wrappers.Monitor(env, './tmp_discrete/cartpole',force=True)

    print('Action space is:', env.action_space)
    print('Observation space is:', env.observation_space)

    # Q-table for the discretized states, and two actions
    num_states = DISCRETE_STEPS ** 4
    qtable = [[0., 0.] for state in range(num_states)]

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        state = make_state(state)

        terminate = False
        cumulative_reward = 0.0

        #TO DO
        
        
if __name__ == '__main__':
    main()