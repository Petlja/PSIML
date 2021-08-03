from gym import wrappers
import os


class EnvWrapper(object):
    def __init__(self, gym_env, steps=None):
        self.env = gym_env
        if steps is not None:
            self.env._max_episode_steps = steps
        wrappers.Monitor(self.env, os.path.join('./tmp_deep_rl/', self.env.spec.id), force=True)

    def state_size(self):
        return self.env.observation_space.shape[0]

    def action_size(self):
        if hasattr(self.env.action_space, 'n'):
            return self.env.action_space.n
        else:
            return self.env.action_space.high.shape[0]

    def action_limit(self):
        if hasattr(self.env.action_space, 'high'):
            return self.env.action_space.high[0]
        else:
            return 1

    #   Displays current state
    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done
