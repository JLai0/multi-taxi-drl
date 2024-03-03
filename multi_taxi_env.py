import argparse
from dataclasses import dataclass
import logging
import random
from contextlib import closing
from io import StringIO
from os import path
from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym

from commons import create_env_map, setup_logger

env_logger = setup_logger('env_logger', './logs/environment.log', logging.WARNING)


@dataclass
class Location:
    x: int = 0
    y: int = 0


def get_random_location(num_columns=5, num_rows=5):
    return int(np.random.randint(num_columns)), int(np.random.randint(num_rows))


@dataclass
class Taxi:
    loc: Location
    pass_idx: int = -1  # -1 refers to no passenger on board, rxefers to index in passenger list

@dataclass
class Passenger:
    loc: Location  # At beginning this is the start location of a passenger and later his/her current location
    dest: Location
    served: bool = False

    def loc_equals_dest(self):
        return self.loc == self.dest

@dataclass
class VerticalObstacle:
    loc: Location
    move_prob: float = 0.5
    direction: int = 0 # 0 = Down, 1 = Up

@dataclass
class HorizontalObstacle:
    loc: Location
    move_prob: float = 0.5
    direction: int = 0 # 0 = Left, 1 = Right


@dataclass
class State:
    taxis: list[Taxi] = field(default_factory=list)
    passengers: list[Passenger] = field(default_factory=list)

    def flatten(self):
        result = []
        [(result.append(t.loc.x), result.append(t.loc.y)) for t in self.taxis]
        [(result.append(p.loc.x), result.append(p.loc.y), result.append(p.dest.x), result.append(p.dest.y)) for p in
         self.passengers]
        return result


written_actions = ['South', 'North', 'East', 'West', 'Wait', 'Pickup', 'Dropoff']


class MultiTaxiEnv(gym.Env):
    """
    A modified version of the taxi problem  
    from the paper "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition" 
    by Tom Dietterich 
    """

    def __init__(self, args, env_map, training=True):
        env_logger.info(f'{id(self)} Initializing the environment.')
        self.number_of_rows = args.number_of_rows
        self.number_of_columns = args.number_of_columns
        self.env_map = np.asarray(env_map, dtype="c")
        self.s = State()
        self.s.taxis = [Taxi(Location()) for i in range(args.number_of_taxis)]
        # self.number_of_obstacles = args.number_of_obstacles  # Add this argument to your argparse
        self.vobstacles = [VerticalObstacle(Location(*get_random_location(self.number_of_columns, self.number_of_rows))) for _ in range(args.number_of_vobstacles)]
        self.hobstacles = [HorizontalObstacle(Location(*get_random_location(self.number_of_columns, self.number_of_rows))) for _ in range(args.number_of_hobstacles)]
        self.s.passengers = [Passenger(Location(), Location()) for i in range(args.number_of_passengers)]

        self.action_space = gym.spaces.Discrete(len(written_actions) * args.number_of_taxis)
        self.observation_space = args.number_of_taxis * 3 + args.number_of_passengers * 4
        self.passenger_colors = [';'.join([str(random.randint(0, 255)) for i in range(3)]) for p in self.s.passengers]
        if args.number_of_passengers < 6:
            """num = 255
            self.taxi_colors = []
            for i in range(args.number_of_taxis):
                self.taxi_colors.append(f'255;{num};0')
                num -= 40"""
            possible_passenger_colors = ['0;255;205', '0;205;255', '0;155;255', '0;105;255', '0;55;255']
            self.passenger_colors = possible_passenger_colors[:args.number_of_passengers]
        else:
            print("error: no more than 5 passengers allowed")
        if args.number_of_taxis < 6:
            possible_taxi_colors = ['255;255;0', '255;215;0', '255;175;0', '255;135;0', '255;95;0']
            self.taxi_colors = possible_taxi_colors[:args.number_of_taxis]
        else:
            print("error: no more than 5 taxis allowed")

    def is_cell_occupied(self, y, x):
        # Check if the location is occupied by a taxi
        for taxi in self.s.taxis:  # Assuming self.taxis is a list of taxis in the environment
            if taxi.loc.y == y and taxi.loc.x == x:
                return True
        # Extend this logic to include other entities, such as obstacles
        for obstacle in self.vobstacles:  # Assuming self.vobstacles tracks obstacles
            if obstacle.loc.y == y and obstacle.loc.x == x:
                return True # Assuming self.vobstacles tracks obstacles
        for obstacle in self.hobstacles:  # Assuming self.vobstacles tracks obstacles
            if obstacle.loc.y == y and obstacle.loc.x == x:
                return True # Assuming self.vobstacles tracks obstacles
        for passenger in self.s.passengers:
            if passenger.loc.y == y and passenger.loc.x == x:
                return True
        # Add similar checks for any other types of entities you have
        return False

    def step(self, actions: list[int]):
        terminated, truncated = self._all_passengers_served(), False

        if terminated:  # Terminated
            # TODO: Normally, we shouldn't reach this, right?
            return self.s, 0, terminated, truncated, {} 

        # Check for truncated
        if self.remaining_steps > 0:
            self.remaining_steps -= 1
        else:  # No time left
            truncated = True
            env_logger.info(f'{id(self)} No steps remaining.')
            return self.s, 0, terminated, truncated, {}

        if not isinstance(actions, list):
            actions = [actions]
            # TODO: actions = [actions % 7, actions // 7]  # TODO: Works only for two agents

        for obstacle in self.vobstacles:
                next_y = obstacle.loc.y - 1 if obstacle.direction == 1 else obstacle.loc.y + 1
                
                # Check for boundary collisions
                if next_y < 0 or next_y >= self.number_of_rows:
                    obstacle.direction = 0 if obstacle.direction == 1 else 1
                    continue  # Skip the rest of the loop and don't move this obstacle this turn
                
                # Check for collisions with other entities
                if self.is_cell_occupied(next_y, obstacle.loc.x):
                    obstacle.direction = 0 if obstacle.direction == 1 else 1
                    continue  # Skip moving this obstacle due to collision
                
                # If no collision, update position
                obstacle.loc.y = next_y
        
        for obstacle in self.hobstacles:
                next_x = obstacle.loc.x - 1 if obstacle.direction == 0 else obstacle.loc.x + 1
                
                # Check for boundary collisions
                if next_x < 0 or next_x >= self.number_of_columns:
                    obstacle.direction = 1 if obstacle.direction == 0 else 0
                    continue  # Skip the rest of the loop and don't move this obstacle this turn
                
                # Check for collisions with other entities
                if self.is_cell_occupied(obstacle.loc.y, next_x):
                    obstacle.direction = 1 if obstacle.direction == 0 else 0
                    continue  # Skip moving this obstacle due to collision
                
                # If no collision, update position
                obstacle.loc.x = next_x

        rewards = [-1]*len(self.s.taxis)  # default reward when there is no pickup/dropoff but a movement
        taxis_and_id_in_random_order = random.sample(list(zip(list(range(len(self.s.taxis))), self.s.taxis)), len(self.s.taxis))
        for (taxi_idx, taxi) in taxis_and_id_in_random_order:
            if actions[taxi_idx] < 4:  # Attempt to move
                next_loc = self._get_next_location(taxi, actions[taxi_idx])

                if self._location_occupied_by_taxi(taxi_idx, next_loc):
                    taxi.loc = next_loc
                    if taxi.pass_idx >= 0:
                        self.s.passengers[taxi.pass_idx].loc.x = next_loc.x
                        self.s.passengers[taxi.pass_idx].loc.y = next_loc.y

            elif actions[taxi_idx] == 4:  # wait
                rewards[taxi_idx] = -1  # TODO: Find out which value is good

            elif actions[taxi_idx] == 5:  # pickup
                passenger_idx = self._taxi_is_at_pass_loc(taxi.loc)
                if taxi.pass_idx < 0 and passenger_idx is not None:
                    taxi.pass_idx = passenger_idx
                else:
                    rewards[taxi_idx] = -10  # Illegal pickup

            elif actions[taxi_idx] == 6:  # dropoff
                if taxi.pass_idx >= 0 and self._taxi_is_at_dest(taxi):  # Pass loc now equals dest
                    self.s.passengers[taxi.pass_idx].loc.x = taxi.loc.x
                    self.s.passengers[taxi.pass_idx].loc.y = taxi.loc.y
                    self.s.passengers[taxi.pass_idx].served = True
                    taxi.pass_idx = -1
                    terminated = self._all_passengers_served()
                    rewards[taxi_idx] = 40
                else:  # dropoff at wrong location
                    rewards[taxi_idx] = -10

        env_logger.info(f'{id(self)} Actions: {actions}; {written_actions[actions[0]]}; rewards: {rewards}; terminated: {terminated}; truncated: {truncated}; remaining steps: {self.remaining_steps}')

        self.last_actions = actions
        if terminated:
            env_logger.warning(f'{id(self)} @{self.remaining_steps} all passengers served; reward: {rewards}')
        env_logger.info(f'{id(self)} Step to: {self.s}')
        # return (self.s, np.asarray(rewards).sum(), terminated, truncated, {"action_mask": self._get_action_mask()})  # TODO: Why doesn't it work with tianshou?
        return (self.s, np.asarray(rewards).sum(), terminated, truncated, {})
    

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # Needed to seed self.np_random
        try_count = 0
        while not self._is_starting_state(self.s) or try_count == 0:
            try_count += 1
            nof_locations = len(self.s.taxis) + len(self.s.passengers) * 2
            cs = np.random.randint(low=self.number_of_columns, size=nof_locations)
            rs = np.random.randint(low=self.number_of_rows, size=nof_locations)
            locations = np.stack((cs, rs), axis=1)

            counter = 0
            for i in range(len(self.s.taxis)):
                self.s.taxis[i].loc.y, self.s.taxis[i].loc.x = locations[counter, 0], locations[counter, 1]
                counter += 1
            for i in range(len(self.s.passengers)):
                self.s.passengers[i].loc.y, self.s.passengers[i].loc.x = locations[counter, 0], locations[counter, 1]
                self.s.passengers[i].dest.y, self.s.passengers[i].dest.x = locations[counter + 1, 0], locations[counter + 1, 1]
                self.s.passengers[i].served = False
                counter += 2

            if try_count > 100:
                env_logger.info(f'{id(self)} Try count: {try_count}')

        self.remaining_steps, self.last_actions = 150, None
        env_logger.info(f'{id(self)} Reset: {self.s}')
        # return self.s, {"action_mask": self._get_action_mask()}  # TODO: Why doesn't it work with tianshou?
        return self.s, {}

    def render(self):
        desc = self.env_map.copy().tolist()
        outfile = StringIO()
        out = [[c.decode("utf-8") for c in line] for line in desc]  # create 2D array from list

        def ul(x):
            return "_" if x == " " else x

        for passenger_idx, passenger in enumerate(self.s.passengers):
            if passenger.served is False:
                passenger_color = self.passenger_colors[passenger_idx]
                out[1 + passenger.loc.y][2 * passenger.loc.x + 1] = self._colorize(chr(97 + passenger_idx),
                                                                                   passenger_color)
                out[1 + passenger.dest.y][2 * passenger.dest.x + 1] = self._colorize(chr(65 + passenger_idx),
                                                                                     passenger_color)

        for taxi_idx, taxi in enumerate(self.s.taxis):
            taxi_color = self.taxi_colors[taxi_idx]
            if taxi.pass_idx < 0:  # No passenger on board of the taxi
                taxi_on_pass = None
                for p in self.s.passengers:
                    if p.loc == taxi.loc:
                        taxi_on_pass = p
                        break
                if taxi_on_pass is None:
                    out[1 + taxi.loc.y][2 * taxi.loc.x + 1] = self._colorize(str(taxi_idx), taxi_color)  # taxi color
                else:
                    out[1 + taxi.loc.y][2 * taxi.loc.x + 1] = self._colorize(chr(97 + self.s.passengers.index(p)),
                                                                             taxi_color)  # taxi color + pass letter
            else:  # passenger in taxi
                out[1 + taxi.loc.y][2 * taxi.loc.x + 1] = self._colorize(str(taxi_idx), self.passenger_colors[
                    taxi.pass_idx])  # pass color + taxi idx
        for obstacle in self.vobstacles:
            out[obstacle.loc.y + 1][obstacle.loc.x * 2 + 1] = 'Y'
        for obstacle in self.hobstacles:
            out[obstacle.loc.y + 1][obstacle.loc.x * 2 + 1] = 'X'

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        # TODO: Clarify rendering
        # Idea 1: Give taxis different colors(, add a corresponding legend), don't give them the number, and add letter of passenger when passenger inside 
        # Idea 2: Taxis gets color of passenger
        # Idea 3: Combine 1 and 2 to make passengers behind taxis that are not in a taxi still visible

        # TODO
        # if self.last_actions is not None:
        #     outfile.write(f"({', '.join([written_actions[a] for a in self.last_actions])})\n")
        # else:
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def _all_passengers_served(self):
        return all([p.loc_equals_dest() for p in self.s.passengers]) and all([t.pass_idx == -1 for t in self.s.taxis])

    def _pass_is_at_dest(self, passenger):
        return passenger.loc == passenger.dest

    def _taxi_is_at_pass_loc(self, taxi_loc):
        for passenger_idx, passenger in enumerate(self.s.passengers):
            if passenger.served == False and taxi_loc == passenger.loc:
                return passenger_idx
        return None

    def _taxi_is_at_dest(self, taxi):
        return taxi.loc == self.s.passengers[taxi.pass_idx].dest

    def _is_free(self, x, y):
        for t in self.s.taxis:
            if t.loc.x == x and t.loc.y == y:
                return False
        return True

    def _get_next_location(self, taxi, action):
        next_loc = Location(taxi.loc.x, taxi.loc.y)
        if action == 0:  # move down
            next_loc.y = min(taxi.loc.y + 1, self.number_of_rows - 1)
        elif action == 1:  # move up
            next_loc.y = max(taxi.loc.y - 1, 0)
        if action == 2 and self.env_map[1 + taxi.loc.y, 2 * taxi.loc.x + 2] == b":":  # move right
            next_loc.x = min(taxi.loc.x + 1, self.number_of_columns - 1)
        elif action == 3 and self.env_map[1 + taxi.loc.y, 2 * taxi.loc.x] == b":":  # move left
            next_loc.x = max(taxi.loc.x - 1, 0)
        return next_loc

    def _location_occupied_by_taxi(self, taxi_idx, next_loc):
        taxi_locations = [t.loc for t in self.s.taxis]
        taxi_locations.pop(taxi_idx)
        return next_loc not in taxi_locations

    def _colorize(self, string: str, color: str) -> str:
        """Returns string surrounded by appropriate terminal colour codes to print colourised text.

        Args:
            string: The message to colourise
            color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
            bold: If to bold the string
            highlight: If to highlight the string

        Returns:
            Colourised string
        """
        attr = []
        if string.isnumeric():
            attr.append('48;2;' + color)  # Background color
        else:
            attr.append('38;2;' + color)  # Text color
        attrs = ";".join(attr)
        return f"\x1b[{attrs}m{string}\x1b[0m"

    def _create_map(self, walls=.0):
        # How to ensure that there is no unpassable border?
        # places for unpassable borders:
        # def. border: line of walls that form a straight longer wall
        # - DONE: line: vertical line of walls that goes all the way through the environment
        # TODO:
        # - corner: two orthogonal borders
        # - box: 4 borders forming a square
        # both are similar
        # ideas:
        # - check everything with for loops -> not very efficient

        first_line = '+' + (self.number_of_columns - 1) * '--' + '-+'
        map = [first_line]
        for r in range(self.number_of_rows):
            str = '|'
            all_walls_set_in_column = [True] * self.number_of_columns
            # approach for preventing lines:
            # check at the end of each row and if all other walls are set, the last one can't be set
            for c in range(self.number_of_columns - 1):  # -1 bc last one isn't random
                if (r == self.number_of_rows - 1 and all_walls_set_in_column[c] is True) or random.random() > walls:
                    str += ' :'
                    all_walls_set_in_column[c] = False
                else:
                    str += ' |'
            str += ' |'
            map.append(str)
        map.append(first_line)
        return map

    @staticmethod
    def _is_starting_state(state):
        locations = []
        [locations.append((obj.loc.x, obj.loc.y)) for obj in state.taxis + state.passengers]
        [locations.append((obj.dest.x, obj.dest.y)) for obj in state.passengers]
        no_loc_equals = len(set(locations)) == len(locations)
        return no_loc_equals
        # locations = []
        # [locations.append((obj.loc.x, obj.loc.y)) for obj in self.s.taxis + self.s.passengers]
        # no_loc_equals = len(set(locations)) == len(locations)  # Two taxis not at the same location, two passengers not at the same location, taxi and passenger not at the same location
        # no_loc_equals_dest = not (any([passenger.loc == passenger.dest for passenger in self.s.passengers]))
        # return no_loc_equals and no_loc_equals_dest

    def _get_action_mask(self):
        """Computes an action mask for the action space using the state information."""
        masks = []
        for taxi in self.s.taxis:
            taxi_x, taxi_y, pass_in_taxi = taxi.loc.x, taxi.loc.y, taxi.pass_idx >= 0
            mask = np.zeros(self.action_space.n, dtype=np.int8)
            mask[4] = 1  # Waiting is always allowed

            if taxi_y < self.number_of_rows - 1 and self._is_free(taxi_x, taxi_y + 1):  # move south
                mask[0] = 1
            if taxi_y > 0 and self._is_free(taxi_x, taxi_y - 1):  # move north
                mask[1] = 1
            if taxi_x < self.number_of_columns - 1 and self.env_map[taxi_y + 1, 2 * taxi_x + 2] == b":" \
                    and self._is_free(taxi_x + 1, taxi_y):  # move east
                mask[2] = 1
            if taxi_x > 0 and self.env_map[taxi_y + 1, 2 * taxi_x] == b":" and self._is_free(taxi_x - 1, taxi_y):  # move west
                mask[3] = 1
            if (not pass_in_taxi) and (self._taxi_is_at_pass_loc(taxi.loc) is not None):  # pick up
                mask[4] = 1
            if pass_in_taxi and (self._taxi_is_at_dest(taxi)):  # dropoff
                mask[5] = 1

            masks.append(mask)
        return masks


def main():
    random_seed = 1
    random.seed(random_seed), np.random.seed(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_rows', help='Number of rows.', type=int, default=5)
    parser.add_argument('--number_of_columns', help='Number of columns.', type=int, default=5)
    parser.add_argument('-ab', '--amount_of_borders', type=float, help='The amount of borders in the environment.', default=0.0)
    parser.add_argument('--number_of_taxis', help='Number of taxis.', type=int, default=1)
    parser.add_argument('--number_of_passengers', help='Number of passengers.', type=int, default=1)
    parser.add_argument('--number_of_vobstacles', help='Number of vertical obstacles.', type=int, default=1)
    parser.add_argument('--number_of_hobstacles', help='Number of horizontal obstacles.', type=int, default=1)
    args = parser.parse_args()

    env_map = create_env_map(args.number_of_columns, args.number_of_rows, args.amount_of_borders)

    for i in range(100):
        max_nof_steps, counter, terminated, truncated = 160, 0, False, False
        env = MultiTaxiEnv(args, env_map)
        obs, _ = env.reset()
        print(env.render())

        while (not terminated) and (not truncated) and (counter < max_nof_steps):
            counter += 1
            actions = [random.randint(0, 6) for i in range(args.number_of_taxis)]
            print(actions)
            obs, reward, terminated, truncated, info = env.step(actions)
            
            print(
                f"[{counter}:{max_nof_steps}] reward: {reward}; actions: {[written_actions[a] for a in actions]}; terminated: {terminated}; truncated: {truncated}")
            print(env.render())

        # TODO: Why don't we see a dropoff?
        # TODO: Check which method the agent learned - bcs it would make sense to simply stay at a location.
        # TODO: Add documentation of functions and add when possible input and output types or arguments to functions.

if __name__ == "__main__":
    main()
