import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, storage_size, obs_dtype, act_dtype, default_dtype):
        self._data = []
        self._max_len = int(storage_size)

        self._o_dtype = obs_dtype
        self._a_dtype = act_dtype
        self._def_dtype = default_dtype
        self._next_id = 0
        self._write_flg = True
        np.random.seed(123)

    @staticmethod
    def _preprocess(var, dtype):
        if torch.is_tensor(var):
            var = var.detach().numpy()
        if hasattr(var, 'shape'):
            var = var.astype(dtype)
            var = np.squeeze(var)
        else:
            var = dtype(var)

        return var

    def add(self, obs, act, next_obs, rew, done):

        obs = self._preprocess(var=obs, dtype=self._o_dtype)
        act = self._preprocess(var=act, dtype=self._a_dtype)
        next_obs = self._preprocess(var=next_obs, dtype=self._o_dtype)
        rew = self._preprocess(var=rew, dtype=self._def_dtype)
        done = self._preprocess(var=done, dtype=self._def_dtype)

        if len(self._data) < self._max_len:
            self._data.append((obs, act, next_obs, rew, done))
        else:
            self._data[self._next_id] = (obs, act, next_obs, rew, done)

        self._next_id = (self._next_id + 1) % self._max_len

    def sample(self, batch_size):
        cap = len(self._data)
        if cap < batch_size:
            replace = True
        else:
            replace = False
        idxs = list(np.random.choice(cap, batch_size, replace=replace))
        obss, actions, next_obss, rewards, dones = [], [], [], [], []

        for idx in idxs:
            data = self._data[idx]
            obss.append(data[0])
            actions.append(data[1])
            next_obss.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])

        obss = np.asarray(obss).reshape((batch_size, -1))
        actions = np.asarray(actions).reshape((batch_size, -1))
        next_obss = np.asarray(next_obss).reshape((batch_size, -1))
        rewards = np.asarray(rewards).reshape((batch_size, -1))
        dones = np.asarray(dones).reshape((batch_size, -1))

        return torch.from_numpy(obss), torch.from_numpy(actions), torch.from_numpy(next_obss), \
               torch.from_numpy(rewards), torch.from_numpy(dones)