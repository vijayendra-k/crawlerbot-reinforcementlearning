"""
@author: Ju Shen
@email: jshen1@udayton.edu
@date: 02-16-2023
"""
import random
import numpy as np
import math as mth


# The state class
class State:
    def __init__(self, angle1=0, angle2=0):
        self.angle1 = angle1
        self.angle2 = angle2


class ReinforceLearning:

    #
    def __init__(self, unit=5):

        ####################################  Needed: here are the variable to use  ################################################

        # The crawler agent
        self.crawler = 0

        # Number of iterations for learning
        self.steps = 1000

        # learning rate alpha
        self.alpha = 0.2

        # Discounting factor
        self.gamma = 0.95

        # E-greedy probability
        self.epsilon = 0.1

        self.Qvalue = []  # Update Q values here
        self.unit = unit  # 5-degrees
        self.angle1_range = [-35, 55]  # specify the range of "angle1"
        self.angle2_range = [0, 180]  # specify the range of "angle2"
        self.rows = int(1 + (self.angle1_range[1] - self.angle1_range[0]) / unit)  # the number of possible angle 1
        self.cols = int(1 + (self.angle2_range[1] - self.angle2_range[0]) / unit)  # the number of possible angle 2

        ########################################################  End of Needed  ################################################

        self.pi = []  # store policies
        self.actions = [-1, +1]  # possible actions for each angle

        # Controlling Process
        self.learned = False

        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)

        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))

        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)

    # Reset the learner to empty
    def reset(self):
        self.Qvalue = []  # store Q values
        self.R = []  # not working
        self.pi = []  # store policies

        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)

        # Initiliaize all the Reward
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.R.append(row)

        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))

        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)

        # Controlling Process
        self.learned = False

    # Set the basic info about the robot
    def setBot(self, crawler):
        self.crawler = crawler

    def storeCurrentStatus(self):
        self.old_location = self.crawler.location
        self.old_angle1 = self.crawler.angle1
        self.old_angle2 = self.crawler.angle2
        self.old_contact = self.crawler.contact
        self.old_contact_pt = self.crawler.contact_pt
        self.old_location = self.crawler.location
        self.old_p1 = self.crawler.p1
        self.old_p2 = self.crawler.p2
        self.old_p3 = self.crawler.p3
        self.old_p4 = self.crawler.p4
        self.old_p5 = self.crawler.p5
        self.old_p6 = self.crawler.p6

    def reverseStatus(self):
        self.crawler.location = self.old_location
        self.crawler.angle1 = self.old_angle1
        self.crawler.angle2 = self.old_angle2
        self.crawler.contact = self.old_contact
        self.crawler.contact_pt = self.old_contact_pt
        self.crawler.location = self.old_location
        self.crawler.p1 = self.old_p1
        self.crawler.p2 = self.old_p2
        self.crawler.p3 = self.old_p3
        self.crawler.p4 = self.old_p4
        self.crawler.p5 = self.old_p5
        self.crawler.p6 = self.old_p6

    def updatePolicy(self):
        # After convergence, generate policy y
        for r in range(self.rows):
            for c in range(self.cols):
                max_idx = 0
                max_value = -1000
                for i in range(9):
                    if self.Qvalue[r][9 * c + i] >= max_value:
                        max_value = self.Qvalue[r][9 * c + i]
                        max_idx = i

                # Assign the best action
                self.pi[r][c] = max_idx

    # This function will do additional saving of current states than Q-learning
    def onLearningProxy(self, option):
        self.storeCurrentStatus()
        if option == 0:
            self.onMonteCarlo()
        elif option == 1:
            self.onTDLearning()
        elif option == 2:
            self.onQLearning()
        self.reverseStatus()

        # Turn off learned
        self.learned = True

    # For the play_btn uses: choose an action based on the policy pi
    def onPlay(self, ang1, ang2, mode=1):

        epsilon = self.epsilon

        ang1_cur = ang1
        ang2_cur = ang2

        # get the state index
        r = int((ang1_cur - self.angle1_range[0]) / self.unit)
        c = int((ang2_cur - self.angle2_range[0]) / self.unit)

        # Choose an action and udpate the angles
        idx, angle1_update, angle2_update = self.chooseAction(r=r, c=c)
        ang1_cur += self.unit * angle1_update
        ang2_cur += self.unit * angle2_update

        return ang1_cur, ang2_cur

    ####################################  Needed: here are the functions you need to use  ################################################

    # This function is similar to the "runReward()" function but without returning a reward.
    # It only update the robot position with the new input "angle1" and "angle2"
    def setBotAngles(self, ang1, ang2):
        self.crawler.angle1 = ang1
        self.crawler.angle2 = ang2
        self.crawler.posConfig()

    # Method 1: You don't need to implement this function
    def onMonteCarlo(self):
        # You need to implement this function for the project 4 part 1
        return

    # Method 2: You don't need to implement this function
    def onTDLearning(self):
        # You don't have to work on it for the moment
        return

    # Given the current state, return an action index and angle1_update, angle2_update
    # Return valuse
    #  - index: any number from 0 to 8, which indicates the next action to take, according to the e-greedy algorithm
    #  - angle1_update: return the angle1 new value according to the action index, one of -1, 0, +1
    #  - angle2_update: the same as angle1

    def chooseAction(self, r, c):
        current_state_qvalues = self.Qvalue[r][(9 * c): (9 * c) + 9]
        max_value = max(current_state_qvalues)
        min_value = min(current_state_qvalues)

        if max_value == min_value:
            possible_moves = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
            angle1_update, angle2_update = random.choice(possible_moves)
            idx = self.actions.index((angle1_update, angle2_update))
        else:
            exploration_count = int(self.epsilon * 100)
            exploitation_count = int((1 - self.epsilon) / 8 * 100)

            max_indices = []
            for i in range(len(current_state_qvalues)):
                if current_state_qvalues[i] == max_value:
                    max_indices.append(i)

            max_index = random.choice(max_indices)

            e_greedy_list = [max_index] * exploration_count
            for i in range(8):
                if i != max_index:
                    e_greedy_list.extend([i] * exploitation_count)

            idx = random.choice(e_greedy_list)
            angle1_update, angle2_update = self.actions[idx]

        angle1_candidate = angle1_update * self.unit + self.crawler.angle1
        angle2_candidate = angle2_update * self.unit + self.crawler.angle2

        if not self.angle1_range[0] <= angle1_candidate <= self.angle1_range[1]:
            angle1_update = 0
            idx = self.actions.index((angle1_update, angle2_update))

        if not self.angle2_range[0] <= angle2_candidate <= self.angle2_range[1]:
            angle2_update = 0
            idx = self.actions.index((angle1_update, angle2_update))

        return idx, angle1_update, angle2_update
    # Method 3: Bellman operator based Q-learning

    def onQLearning(self):
        for step in range(self.steps):
            current_angle1_idx = (self.crawler.angle1 - self.angle1_range[0]) // self.unit
            current_angle2_idx = (self.crawler.angle2 - self.angle2_range[0]) // self.unit

            old_location = self.crawler.location[0]

            action_idx, angle1_update, angle2_update = self.chooseAction(current_angle1_idx, current_angle2_idx)

            new_angle1 = self.crawler.angle1 + (angle1_update * self.unit)
            new_angle2 = self.crawler.angle2 + (angle2_update * self.unit)

            self.setBotAngles(new_angle1, new_angle2)

            new_location = self.crawler.location[0]
            reward = new_location - old_location

            next_angle1_idx = current_angle1_idx + angle1_update
            next_angle2_idx = current_angle2_idx + angle2_update

            if next_angle1_idx < 0:
                next_angle1_idx = 0
            elif next_angle1_idx >= len(self.Qvalue):
                next_angle1_idx = len(self.Qvalue) - 1

            if next_angle2_idx < 0:
                next_angle2_idx = 0
            elif next_angle2_idx >= len(self.Qvalue[0]) // 9:
                next_angle2_idx = len(self.Qvalue[0]) // 9 - 1

            q_value = self.Qvalue[current_angle1_idx][9 * current_angle2_idx + action_idx]
            max_new_loc = max(self.Qvalue[next_angle1_idx][9 * next_angle2_idx: 9 * next_angle2_idx + 9])
            new_q_value = int((1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * max_new_loc))
            self.Qvalue[current_angle1_idx][9 * current_angle2_idx + action_idx] = new_q_value

        return
