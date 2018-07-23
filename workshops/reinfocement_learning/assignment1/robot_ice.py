import argparse
import itertools
import os
import sys
import time
import random
import value_iteration

# 0: Normal ice
# 1: Cracked ice
# 2: Wrecked ship (treasure)
# 3: Goal

SQUARE_SIZE = 4

world = [[0,0,0,3],
         [0,1,0,1],
         [0,0,2,1],
         [0,1,1,1]]

world_column = lambda i: enumerate(reversed([row[i] for row in world]))

pSlip = 0.05

# Model

def getTransitionProbability(coord1, action, coord2):
    """
    Given two coordinates and an action, return the transition probability.
    Parameters
    ----------
    coord1: tuple of int
        Two element tuple containing the current coordinates of the robot.
    action: str
        String representing the robot action, options are i
        'up', 'down', 'left', 'right'.
    coord2: tuple of int
        Two element tuple containing the next coordinates of the robot.
    """
    # We check where the robot is
    x1, y1 = coord1
    x2, y2 = coord2

    # We compute the distances traveled by the robot
    robotMovementX = x2 - x1
    robotMovementY = y2 - y1

    # The robot can only move in a single direction
    if robotMovementX * robotMovementY != 0:
        return 0.0

    # Remove cases where the robot cannot possibly move
    if ((action == 'up' and y1 == 3) or
        (action == 'down' and y1 == 0) or
        (action == 'left' and x1 == 0) or
        (action == 'right' and x1 == 3) or
        (world[3-y1][x1] == 3) or
        (world[3-y1][x1] == 1)):
        if robotMovementX + robotMovementY == 0:
            return 1.0
        return 0.0

    if action == 'up':
        if robotMovementX != 0:
            return 0.0
        next_crack = next((i for i,v in world_column(x1) if v == 1 and i >= y1), None)
        # If we arrived at the wall, or at the next crack
        if y2 == next_crack or (y2 == 3 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementY == 1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementY == 1:
            return 1 - pSlip
        return 0.0

    if action == 'down':
        if robotMovementX != 0:
            return 0.0
        next_crack = next((i for i,v in reversed(list(world_column(x1))) if v == 1 and i <= y1), None)
        if y2 == next_crack or (y2 == 0 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementY == -1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementY == -1:
            return 1 - pSlip
        return 0.0

    if action == 'left':
        if robotMovementY != 0:
            return 0.0
        next_crack = next((i for i,v in reversed(list(enumerate(world[3-y1]))) if v == 1 and i <= x1), None)
        if x2 == next_crack or (x2 == 0 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementX == -1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementX == -1:
            return 1 - pSlip
        return 0.0

    if action == 'right':
        if robotMovementY != 0:
            return 0.0
        next_crack = next((i for i,v in enumerate(world[3-y1]) if v == 1 and i >= x1), None)
        if x2 == next_crack or (x2 == 3 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementX == 1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementX == 1:
            return 1 - pSlip
        return 0.0


def getReward(coord1, action, coord2):
    """
    Given a state, return the reward.
    Parameters
    ----------
    coord1: tuple of int
        Two element tuple containing the current coordinates of the robot.
    action: str
        String representing the robot action, options are i
        'up', 'down', 'left', 'right'.
    coord2: tuple of int
        Two element tuple containing the next coordinates of the robot.
    Returns
    -------
    reward: float
        If the robot reaches the treasure, reward is +20. Reaching the goal is
        +100. Stepping onto cracked ice is -10.
    """
    #if coord1 == coord2:
    #    return 0.0

    x, y = coord2

    if world[3-y][x] == 3:
        return 100.0
    if world[3-y][x] == 2:
        return 20.0
    if world[3-y][x] == 1:
        return -10.0

    return 0.0


def encodeState(coord):
    """
    Convert from coordinate to state_index.
    Parameters
    ----------
    coord: tuple of int
        Two element tuple containing the position of the robot
    Returns
    -------
    state: int
        Index of the state.
    """
    state = 0
    multiplier = 1
    for c in coord:
        state += multiplier * c
        multiplier *= SQUARE_SIZE

    return state


def decodeState(state):
    """
    Convert from state_index to coordinate.
    Parameters
    ----------
    state: int
        Index of the state.
    Returns
    -------
    coord: tuple of int
        Two element tuple containing the position of the robot
    """
    coord = []
    for _ in range(2):
        c = state % SQUARE_SIZE
        state /= SQUARE_SIZE
        coord.append(c)
    return tuple(coord)


#RENDERING

# Special character to go back up when drawing.
up = list("\033[XA")
# Special character to go back to the beginning of the line.
back = list("\33[2K\r")

def goup(x):
    """ Moves the cursor up by x lines """
    while x > 8:
        up[2] = '9'
        print "".join(up)
        x -= 8

    up[2] = str(x + 1)
    print "".join(up)

def godown(x):
    """ Moves the cursor down by x lines """
    while x:
        print ""
        x -= 1

def printState(coord):
    """
    Draw the grid world.
    - @ represents the robot.
    - T represents the treasure.
    - X represents the cracks in the ice.
    - G represents the goal.
    Parameters
    ----------
    coord: tuple of int
        Two element tuple containing the position of the robot
    """
    r_x, r_y = coord
    for y in range(SQUARE_SIZE-1, -1, -1):
        for x in range(SQUARE_SIZE):
            if (r_x, r_y) == (x, y):
                print "@",
            elif world[3-y][x] == 3:
                print "G",
            elif world[3-y][x] == 2:
                print "T",
            elif world[3-y][x] == 1:
                print "X",
            else:
                print ".",
        print ""

def sampleProbability(vec):
    """
    Returns the id of a random element of vec, assuming vec is a list of
    elements which sum up to 1.0.
    The random element is returned with the same probability of its value in
    the input vector.
    """
    p = random.uniform(0, 1)
    for i, v in enumerate(vec):
        if v > p:
            return i
        p -= v

    return i

def sampleSR(s, a, T):
    s1 = sampleProbability(T[s][a])

    return s1, getReward(decodeState(s), A[a], decodeState(s1))

def isTerminal(coord):
    x, y = coord
    if ((world[3-y][x] == 3) or
        (world[3-y][x] == 1)):
        return True
    return False

# Statespace contains the robot (x, y). Note that
# S = [(r_x, r_y), .. ]
S = list(itertools.product(range(SQUARE_SIZE), repeat=2))

# A = robot actions
A = ['up', 'down', 'left', 'right']

def solve_mdp(horizon, epsilon, discount=0.9):
    """
    Construct the gridworld MDP, and solve it using value iteration. Print the
    best found policy for sample states.
    Returns
    -------
    solution: tuple
        First element is a boolean that indicates whether the method has
        converged. The second element is the value function. The third
        element is the Q-value function, from which a policy can be derived.
    """
    print time.strftime("%H:%M:%S"), "- Constructing MDP..."

    # T gives the transition probability for every s, a, s' triple.
    # R gives the reward associated with every s, a, s' triple.
    T = []
    R = []
    for state in range(len(S)):
        coord = decodeState(state)
        T.append([[getTransitionProbability(coord, action,
                                            decodeState(next_state))
                   for next_state in range(len(S))] for action in A])
        R.append([[getReward(coord, action, decodeState(next_state))
                   for next_state in range(len(S))] for action in A])
    

    solution_a, solution_v = value_iteration.valueIteration(len(S), len(A), discount, horizon, epsilon, T, R)
    print(solution_v)
    print(solution_a)
    s = 0

    totalReward = 0
    for t in xrange(100):
        printState(decodeState(s))

        if isTerminal(decodeState(s)):
            break

        s1, r = sampleSR(s, solution_a[s], T)

        totalReward += r
        s = s1

        goup(SQUARE_SIZE)

        state = encodeState(coord)

        # Sleep 1 second so the user can see what is happening.
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ho', '--horizon', default=1000000, type=int,
                        help="Horizon parameter for value iteration")
    parser.add_argument('-e', '--epsilon', default=0.001, type=float,
                        help="Epsilon parameter for value iteration")
    parser.add_argument('-d', '--discount', default=0.9, type=float,
                        help="Discount parameter for value iteration")

    args = parser.parse_args()
    solve_mdp(horizon=args.horizon, epsilon=args.epsilon, discount=args.discount)