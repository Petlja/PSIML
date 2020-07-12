import gym
from gym import wrappers


class CartPoleEnv(object):
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        wrappers.Monitor(self.env, './tmp_deep_q_learning/cartpole',force=True)
    
    #    Observation: 
    #    Type: Box(4)
    #    Num	Observation                 Min         Max
    #    0	Cart Position             -4.8            4.8
    #    1	Cart Velocity             -Inf            Inf
    #    2	Pole Angle                 -24°           24°
    #    3	Pole Velocity At Tip      -Inf            Inf

    #    returns number of observable state dimensions
    def state_size(self):
        return self.env.observation_space.shape[0]

    #    Action:
    #    Type: Discrete(2)
    #    Num	Action
    #    0	Push cart to the left
    #    1	Push cart to the right

    #    returns number of possible actions.
    def action_size(self):
        return self.env.action_space.n

    #   Displays current state
    def render(self):
        self.env.render()

    #   Restarts the environment and returns starting state
    def reset(self):
        state = self.env.reset()
        return state

    #   Performs a provided action inside the environment.
    #   Returns:
    #    next_state
    #    normalized reward
    #    flag that determines if environment is in terminal state
    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        reward = reward if not done else -10
        return next_state, reward, done

