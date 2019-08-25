import copy
import pickle
import random

import numpy as np

from car import Car


class RaceTrack:
    """
    This class implements the race track algorithm
    """

    def __init__(self, track, track_name, start_position_x, start_position_y, max_x, max_y):
        """
        This function initializes the Race Track environment by setting the track, start and finish positions
        and setting the car to it's initial state. This function also renders the initial environment to the terminal
        :param track: The grid of the track with # = wall, S = possible start positions, F = possible finish positions,
        and . = open track
        :param start_position_x: The x value of the chosen start position
        :param start_position_y: The y value fo the chosen start position
        """
        # Initialize track
        self.track_max_x = int(max_x)
        self.track_max_y = int(max_y)
        self.track_name = str(track_name)
        print(track_name)
        self.track = list()
        for row in track:
            self.track.append(list(row))
        self.start_state = Car(start_position_x, start_position_y, 0, 0, self.track_max_x, self.track_max_y)
        self.car = Car(start_position_x, start_position_y, 0, 0, self.track_max_x, self.track_max_y)
        self.cost = 0
        self.wall = '#'
        self.finish = 'F'
        self.num_crashes = 0
        self.__set_car_to_start()
        self.render_state()

        # Initialize MDP
        self.value_matrix = dict()
        self.actions = [tuple((0, -1)), tuple((-1, -1)), tuple((-1, 0)),
                        tuple((1, 0)), tuple((0, 1)), tuple((1, 1)), tuple((1, -1)),  tuple((-1, 1))]
        self.transitions = [1, 1, 1, 0.8, 0.8, 0.8, 0.8,  0.8]  # based on probability of acceleration working
        self.rewards = dict()
        self.gamma = 0.9
        self.policy = None

    def run_racetrack_on_value_iteration(self, restart_on_crash=True):
        """
        This function was used to run the race track simulator using the policy created in with the value iteration
        algorithm. It reads the policy in from a file, so that you do not need to rerun the value iteration
        each time.
        """
        # Read in the policy created by value iteration
        print("Reading in policy created by value iteration...")
        with open(f"policy_{self.track_name}", "rb") as file:
            self.policy = pickle.load(file)
            print(self.policy)

        # Run the race track simulation
        print("Running racetrack simulation...")
        finished = False
        self.__set_car_to_start()
        current_movement_x, current_movement_y = self.__choose_action(self.car.get_state())
        while not finished and self.cost < 1000:
            print(f'a = {current_movement_x}, {current_movement_y}')
            print(f"Starting state = {self.car.get_state()}")
            self.car.accelerate(current_movement_x, current_movement_y)
            print(f'Current position = {self.car.position.x}, {self.car.position.y}')
            print(f'Current velocity = {self.car.velocity.x}, {self.car.velocity.y}')
            print(f'COST = {self.cost}')
            if self.car_has_crashed(self.car):
                print('THE CAR HAS CRASHED $$@@##$@$%!!')
                if restart_on_crash:
                    self.__set_car_to_start()
                else:
                    self.car.set_to_last_position()
                self.car.reset_velocity()
                self.num_crashes += 1
                current_movement_x, current_movement_y = self.__choose_action(self.car.get_state())

            if self.car_has_crossed_the_finish(self.car):
                print('CROSSED THE FINISH LINE!!!!!')
                finished = True
            else:
                self.cost += 1

            self.render_state()

    def run_racetrack_on_sarsa(self, episodes, restart_on_crash, epsilon=0.1, learning_rate=0.1):
        print("Running racetrack simulation with SARSA...")
        self.__initialize_states()
        q = self.__init_q()
        for episode in range(episodes):
            print(f"Episode: {episode}")
            self.__set_car_to_start()
            action = self.__epsilon_greedy(q, epsilon, len(self.actions))
            finished = False
            self.cost = 0
            while not finished and self.cost < 20:
                current_state = self.car.get_state()
                reward = self.__get_reward(current_state, action)
                print(f'a = {action}')
                print(f"Current state = {current_state}")
                successor_car = Car(current_state[0], current_state[1],
                                    current_state[2], current_state[3],
                                    self.track_max_x, self.track_max_y)
                successor_car.accelerate(action[0], action[1])
                successor_state = successor_car.get_state()
                print(f'Successor state = {successor_state}')
                print(f'COST = {self.cost}')
                if self.car_has_crashed(successor_car):
                    print('THE CAR HAS CRASHED $$@@##$@$%!!')
                    if restart_on_crash:
                        successor_car.set_state(tuple((self.start_state.position.x,
                                                       self.start_state.position.y,
                                                       0, 0)))
                    else:
                        successor_car.set_to_last_position()

                if self.car_has_crossed_the_finish(successor_car):
                    print('CROSSED THE FINISH LINE!!!!!')
                    finished = True
                else:
                    self.cost += 1

                successor_action = self.__epsilon_greedy(q, epsilon, len(self.actions))
                q[tuple((current_state, action))] = q[tuple((current_state, action))] + \
                                                    learning_rate * \
                                                    (reward + self.gamma *
                                                     q[tuple((successor_state, successor_action))] -
                                                     q[tuple((current_state, action))])
                self.car = successor_car
                action = successor_action

                self.render_state()

    def car_has_crashed(self, car):
        try:
            crashed = True if self.track[car.position.x][car.position.y] == self.wall else False
            return crashed
        except IndexError:
            return True

    def car_has_crossed_the_finish(self, car):
        return True if self.track[car.position.x][car.position.y] == self.finish else False

    def learn_value_iteration(self):
        self.__value_iteration__()

    def render_state(self):
        """
        This function renders the current state of the race track to the screen with @ representing the current
        position of the car
        """
        print(f"Position = ({self.car.position.x}, {self.car.position.y})")
        print(f"Velocity = ({self.car.velocity.x}, {self.car.velocity.y})")
        print(f"Current cost = {self.cost}")
        self.track[self.car.position.x][self.car.position.y] = '@'
        for row in self.track:
            print("".join(row))

    def __set_car_to_start(self):
        print("Moving to Start Position")
        self.car.position = self.start_state.position

    def __value_iteration__(self, epsilon=0.001):
        policy = dict()
        self.__initialize_states()

        # Run value iteration
        converged = False
        max_iterations = 20
        t = 0
        q = dict()
        previous_value_matrix = None
        while not converged and t < max_iterations:
            t += 1
            print(f"Iterating: t = {t}")
            for state, value in self.value_matrix.items():
                for a_index, action in enumerate(self.actions):
                    if previous_value_matrix:
                        q[tuple((state, action))] += self.__get_reward(state, action) + \
                                                     self.gamma * \
                                                     sum(self.__transition(i)
                                                         * previous_value_matrix[self.__apply_action(state, a)]
                                                         for i, a in enumerate(self.actions))
                    else:
                        q[tuple((state, action))] = self.__get_reward(state, action)

                q_for_this_state = {k: v for (k, v) in q.items() if k[0] == state}
                policy[state] = max(q_for_this_state, key=q_for_this_state.get)[1]
                self.value_matrix[state] = q_for_this_state[(state, policy[state])]

            # Calculate convergence
            if previous_value_matrix:
                diff = self.__get_max_difference(self.value_matrix, previous_value_matrix)
                print(f"Max Difference = {diff}")
                if diff < epsilon:
                    print("Converging...")
                    converged = True

            previous_value_matrix = copy.deepcopy(self.value_matrix)

        # Write policy to file
        with open(f"policy_{self.track_name}", "wb") as file:
            pickle.dump(policy, file, protocol=pickle.HIGHEST_PROTOCOL)

        return policy

    def __initialize_states(self):
        # Initialize the value matrix to all zeros
        for r_index, row in enumerate(self.track):
            for c_index, cell in enumerate(row):
                for i in range(-5, 6):
                    for j in range(-5, 6):
                        state = tuple((r_index, c_index, i, j))
                        self.value_matrix[state] = 0.0
                        self.rewards[state] = self.__calculate_reward(c_index, r_index)

    def __calculate_reward(self, c_index, r_index):
        return -1 if self.track[r_index][c_index] == self.wall else \
                (2 if self.track[r_index][c_index] == self.finish else 1)

    @staticmethod
    def __get_max_difference(value_matrix, previous_value_matrix):
        max_difference = 0
        for state, value in value_matrix.items():
            diff = value - previous_value_matrix[state]
            if diff > max_difference:
                max_difference = diff
        return max_difference

    def __transition(self, action):
        return self.transitions[action]

    def __get_reward(self, state, action):
        car = Car(state[0], state[1], state[2], state[3], self.track_max_x, self.track_max_y)
        car.accelerate(action[0], action[1])
        try:
            reward = self.rewards[car.get_state()]
            return reward
        except KeyError:
            return -1  # Went off board

    def __apply_action(self, state, action):
        car = Car(state[0], state[1], state[2], state[3], self.track_max_x, self.track_max_y)
        car.accelerate(action[0], action[1])
        state = car.get_state()
        return state

    def __choose_action(self, state):
        print(f'Choosing action from current state {state}')
        return self.policy[state]

    def __epsilon_greedy(self, q, epsilon, num_actions):
        if np.random.rand() < epsilon:
            q_for_this_state = {k: v for (k, v) in q.items() if k[0] == self.car.get_state()}
            action = max(q_for_this_state, key=q_for_this_state.get)[1]
        else:
            action = self.actions[np.random.randint(0, num_actions)]
        print(f"Chosen action = {action}")
        return action

    def __init_q(self):
        q = dict()
        for state, value in self.value_matrix.items():
            for a_index, action in enumerate(self.actions):
                q[tuple((state, action))] = random.randint(0, 100)

        print(f"Q = {q}")

        return q


