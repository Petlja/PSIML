import random
import torch

SLIP_CHANCE = 0.05

# Gridworld. ' ' is an ice cell, '*' a crack, 'T' the treasure, and 'G' the goal.
GRID = [
    [' ', ' ', ' ', 'G'],
    [' ', '*', ' ', '*'],
    [' ', ' ', 'T', '*'],
    [' ', '*', '*', '*'],
]

ACTIONS = ['^', 'v', '<', '>']
REWARD = {' ': -5.0, '*': -10.0, 'T': 20.0, 'G': 100.0}
TERMINAL = {' ': False, '*': True, 'T': False, 'G': True}


class Ice(object):
    def __init__(self, slip_chance=0.05, ice_r=-5.0, crack_r=-10.0, treasure_r=20.0, goal_r=100.0):
        self.reset()
        self.grid = GRID
        self.actions = ACTIONS
        self.terminal = TERMINAL
        self.reward = {' ': ice_r, '*': crack_r, 'T': treasure_r, 'G': goal_r}
        self.slip_chance = slip_chance

        self._x = self._y = None
        _ = self.reset()

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        # Put the agent in the bottom-left corner of the environment
        self._x = 0
        self._y = 3

        return self.current_state()

    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
        """
        assert(action >= 0)
        assert(action <= 3)

        # x and y coordinates of the agent if it slips while executing the action
        slip_x = self._x
        slip_y = self._y

        if action == 0 and self._y > 0:
            # Go up
            self._y -= 1
            slip_y = 0
        elif action == 1 and self._y < 3:
            # Go down
            self._y += 1
            slip_y = 3
        elif action == 2 and self._x > 0:
            # Go left
            self._x -= 1
            slip_x = 0
        elif action == 3 and self._x < 3:
            # Go right
            self._x += 1
            slip_x = 3

        # The agent may slip
        if random.random() < self.slip_chance:
            self._x = slip_x
            self._y = slip_y

        # Return the current state, a reward and whether the episode terminates
        cell = self.grid[self._y][self._x]

        return self.current_state(), self.reward[cell], self.terminal[cell]

    def current_state(self):
        return self._y, self._x

    def action_space(self):
        return 4,

    def env_space(self):
        return 4, 4

    def print_qtable_stats(self, qtable):
        zeros = torch.zeros(size=self.action_space(), dtype=torch.float)
        print('ENV')
        for y in range(self.env_space()[1]):
            print(self.grid[y])
        print('QACTIONS')
        print(self.actions)
        print('QVALUES')
        for x in range(self.env_space()[1]):
            for y in range(self.env_space()[0]):
                print("x,y:{0},{1}, f:{2}, qvalues:{3}".format(x, y, self.grid[y][x], qtable[y, x]))
            print()
        print('GREEDY POLICY')
        for y in range(self.env_space()[0]):
            for x in range(self.env_space()[1]):
                if torch.equal(qtable[y,x], zeros):
                    print('?', end='  ')
                else:
                    print(self.actions[torch.argmax(qtable[y, x])], end='  ')
            print()
