"""
Aufgabe 1: Stromrallye (b)

Dieses Skript ist Teil der Einsendung für die

    2. Runde des
    38. Bundeswettbewerbs Informatik

von

    Florian Rädiker.

Teilnahme-ID: 52570
"""
import itertools
import math
import os
from random import Random
from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np

import aufgabe1a

from util import Vector2, save_svg


FIELD_SIZE = 50  # must be > 1
BATTERY_COUNT = 10
NEW_CHARGE = lambda rand: int(round(rand.triangular(0.5, 11.4999, 4)))
PROBABILITY_TAKE_FASTEST = 0.8
PROBABILITY_MARK_AS_MULTI_VISIT = 0.7
PROBABILITY_DO_MULTI_VISIT = 0.6
CALC_FASTEST_WEIGHT = lambda d: d**2
CALC_NORMAL_WEIGHT = lambda d: d

EXTENDED_FILE_FORMAT = True


class NoNewPositionsError(IndexError):
    pass


def calc_weight(difficulty, is_fastest, fastest_weight, normal_weight):
    if is_fastest:
        return fastest_weight(difficulty)
    return normal_weight(difficulty)


def choose_difficulty(rand: Random, fastest_difficulties: Sequence[int], normal_difficulties: Sequence[int],
                      calc_weight: Callable[[int, bool], int] = partial(calc_weight, fastest_weight=CALC_FASTEST_WEIGHT,
                                                                        normal_weight=CALC_NORMAL_WEIGHT)) -> int:
    if not fastest_difficulties and not normal_difficulties:
        raise NoNewPositionsError
    fastest_difficulties = set(fastest_difficulties)
    all_difficulties = list(fastest_difficulties.union(normal_difficulties))
    weights = [calc_weight(d, d in fastest_difficulties) for d in all_difficulties]
    print("all diff", all_difficulties)
    print("weights ", weights)
    d = rand.choices(all_difficulties, weights)[0]
    print("choice is", d)
    return d


def choose_max_difficulty(_, fastest_difficulties, normal_difficulties):
    try:
        return max(itertools.chain(fastest_difficulties, normal_difficulties))
    except ValueError:
        raise NoNewPositionsError


def choose_next_battery(rand: Random, field, normal_difficulties2positions, fastest_difficulties2positions,
                        probability_do_multi_visit,
                        probability_take_fastest,
                        choose_difficulty: Callable[[Random, Sequence[int], Sequence[int]], int] =
                        choose_difficulty):  # change 'choose_difficulty' to 'choose_max_difficulty' to always select
                                             # the largest difficulty
    print("choose difficulty\ndifficulties normal", list(normal_difficulties2positions.keys()))
    print("difficulties fastest", list(fastest_difficulties2positions.keys()))
    positions = None
    if 255 in fastest_difficulties2positions or 255 in normal_difficulties2positions:
        if rand.random() < probability_do_multi_visit:
            if 255 in fastest_difficulties2positions and \
                    (255 not in normal_difficulties2positions or rand.random() < probability_take_fastest):
                positions = fastest_difficulties2positions[255]
            else:
                positions = normal_difficulties2positions[255]
        else:
            try:
                del fastest_difficulties2positions[255]
            except KeyError:
                pass
            try:
                del normal_difficulties2positions[255]
            except KeyError:
                pass
    elif 255 in normal_difficulties2positions:
        if rand.random() < probability_do_multi_visit:
            positions = normal_difficulties2positions[255]
        else:
            del normal_difficulties2positions[255]
    if positions is None:
        difficulty = choose_difficulty(rand, fastest_difficulties2positions.keys(),
                                       normal_difficulties2positions.keys())
        if difficulty in fastest_difficulties2positions and \
                (difficulty not in normal_difficulties2positions or rand.random() < probability_take_fastest):
            positions = fastest_difficulties2positions[difficulty]
        else:
            positions = normal_difficulties2positions[difficulty]

    min_ = math.inf
    minimum_positions = None
    for pos, way in positions:
        # count of fields which need to become ways in field array
        count = sum(1 if field[p] == 0 else 0 for p in way)
        if count < min_:
            min_ = count
            minimum_positions = [(pos, way)]
        elif count == min_:
            minimum_positions.append((pos, way))
    return rand.choice(minimum_positions)


def generate_stromrallye(size: int, battery_count, rand: Random, get_random_charge: Callable[[Random], int],
                         probability_mark_as_multi_visit: float, probability_do_multi_visit: float,
                         probability_take_fastest: float,
                         choose_next_battery: Optional[Callable] = choose_next_battery):
    def get_next_battery_position(current_pos: Vector2, initial_charge: int):
        """
        Search new battery positions and return one (selected using choose_next_battery)
        :param current_pos: position to search new positions from
        :param initial_charge: charge on current_pos
        :return: (position, way to this position)
        """

        normal_difficulties2positions = {}
        fastest_difficulties2positions = {}

        def add_new_position(pos, way, is_fastest):
            if field[pos] == 0 or difficulties[pos] == 255:  # difficulty == 255 means field is multi-visit field
                difficulty = difficulties[pos]  # difficulty for pos
                if difficulties[pos] != 255:
                    difficulties[pos] += 4 if is_fastest else rand.randint(0, 2)
                difficulties2positions = fastest_difficulties2positions if is_fastest else normal_difficulties2positions
                try:
                    difficulties2positions[difficulty].append((pos, way))
                except KeyError:
                    difficulties2positions[difficulty] = [(pos, way)]

        visited = np.zeros(shape, dtype="uint8")  # 0: not visited, 1: fastest visited, 2: normal visited
        visited[current_pos] = 1
        one_visit_positions = [(next_pos, []) for next_pos in current_pos.neighbor_fields(0, size-1, 0, size-1)
                               if field[next_pos] < 2 or difficulties[next_pos] == 255]
        double_visit_positions = []
        for remaining_charge in range(initial_charge-1,
                                      # visit normal fields to remaining_charge = initial_charge-4 (for even
                                      # initial_charge) or initial_charge-3 (for odd initial_charge)
                                      max(initial_charge+(-5 if initial_charge % 2 == 0 else -4), -1),
                                      -1):
            rand.shuffle(one_visit_positions)
            rand.shuffle(double_visit_positions)
            one_visit_positions.sort(key=lambda pos_and_way: 0 if field[pos_and_way[0]] == 1 else 1)  # positions on ways first
            double_visit_positions.sort(key=lambda pos_and_way: 0 if field[pos_and_way[0]] == 1 else 1)
            next_one_visit_positions = []
            next_double_visit_positions = []
            for pos, old_way_to_pos in one_visit_positions:
                if visited[pos] == 0:
                    visited[pos] = 1
                    if remaining_charge == 0 or (remaining_charge < initial_charge-2 and remaining_charge % 2 == 0):
                        add_new_position(pos, old_way_to_pos, remaining_charge == 0)
                    if difficulties[pos] != 255:
                        way_to_pos = old_way_to_pos + [pos]
                        for next_pos in pos.neighbor_fields(0, size-1, 0, size-1):
                            if field[next_pos] < 2 or difficulties[next_pos] == 255:
                                if visited[next_pos] == 0:
                                    next_one_visit_positions.append((next_pos, way_to_pos))
                                else:
                                    next_double_visit_positions.append((next_pos, way_to_pos))
            for pos, old_way_to_pos in double_visit_positions:
                assert visited[pos] != 0
                if visited[pos] == 1:
                    if pos not in old_way_to_pos:
                        visited[pos] = 2
                        if remaining_charge % 2 == 0 and remaining_charge < initial_charge-2:
                            add_new_position(pos, old_way_to_pos, remaining_charge == 0)
                    if difficulties[pos] != 255:
                        way_to_pos = old_way_to_pos + [pos]
                        for next_pos in pos.neighbor_fields(0, size-1, 0, size-1):
                            if field[next_pos] < 2 or difficulties[next_pos] == 255:
                                next_double_visit_positions.append((next_pos, way_to_pos))
            one_visit_positions = next_one_visit_positions
            double_visit_positions = next_double_visit_positions
        for remaining_charge in range(remaining_charge-1,  # start where previous loop stopped
                                      -1, -1):
            rand.shuffle(one_visit_positions)
            one_visit_positions.sort(key=lambda pos_and_way: 0 if field[pos_and_way[0]] == 1 else 1)  # positions on ways first
            next_one_visit_positions = []
            for pos, old_way_to_pos in one_visit_positions:
                if not visited[pos]:
                    visited[pos] = 1
                    if remaining_charge % 2 == 0:
                        add_new_position(pos, old_way_to_pos, remaining_charge == 0)
                    if difficulties[pos] != 255:
                        way_to_pos = old_way_to_pos + [pos]
                        for next_pos in pos.neighbor_fields(0, size-1, 0, size-1):
                            if field[next_pos] < 2 or difficulties[next_pos] == 255:
                                next_one_visit_positions.append((next_pos, way_to_pos))
            one_visit_positions = next_one_visit_positions

        return choose_next_battery(rand, field, normal_difficulties2positions, fastest_difficulties2positions,
                                   probability_do_multi_visit, probability_take_fastest)

    def get_free_neighbor_fields(pos: Vector2):
        return (p for p in pos.neighbor_fields(0, size-1, 0, size-1) if field[p] < 2)

    shape = (size, size)
    field = np.zeros(shape, dtype="uint8")  # 0: free, 1: way or robot battery, 2: battery
    difficulties = np.empty(shape, dtype=int)
    difficulties.fill(1)

    # robot battery
    pos = Vector2(size//2, size//2)
    charge = get_random_charge(rand)
    print("robot battery charge is", charge)

    batteries = [(pos, charge)]  # first element is robot battery field

    field[pos] = 1

    solution = [pos]

    labels = {}  # labels that will be written to output file

    multi_visit_fields = {}  # pos to charge
    all_multi_visit = []

    old_charge = charge

    old_pos = pos

    battery_num_from_which_charge_originates = 0

    while len(batteries) < battery_count:
        print("\nBATTERY")
        if old_charge > 1 and rand.random() < probability_mark_as_multi_visit:
            print("the field for this battery should be multi-visited")
            # the following battery will be visited multiple times
            multi_visit_charge = rand.randint(1, old_charge - 1)
            print("  reduced charge to search new battery positions from", old_charge, "to",
                  old_charge-multi_visit_charge)
            old_charge -= multi_visit_charge
            is_multi_visit = True
        else:
            is_multi_visit = False
        print("search ways from", old_pos, "with charge", old_charge)
        try:
            pos, way = get_next_battery_position(old_pos, old_charge)
        except NoNewPositionsError:
            print("no free way")
            break
        print("position for battery:", pos)
        solution.extend(way)
        solution.append(pos)
        for way_pos in way:
            field[way_pos] = 1

        old_battery_num_from_which_charge_originates = battery_num_from_which_charge_originates

        if difficulties[pos] == 255:
            # this field is not visited for the first time, instead the charge the robot laid on this field earlier is
            # used
            assert field[pos] != 0
            print("position was visited before")
            charge, battery_num_from_which_charge_originates = multi_visit_fields[pos]
            print("  charge on this field is", charge)
            del multi_visit_fields[pos]
            difficulties[pos] = 0
        else:
            charge = get_random_charge(rand)
            # a battery with charge 'charge' is placed at pos
            print("  set charge on this field to", charge)
            assert field[pos] == 0
            field[pos] = 2
            battery_num_from_which_charge_originates = len(batteries)
            batteries.append((pos, charge))

        if is_multi_visit:
            assert pos not in multi_visit_fields
            multi_visit_fields[pos] = (multi_visit_charge, old_battery_num_from_which_charge_originates)
            difficulties[pos] = 255
            print("this field should be multi-visited, charge is going to be", multi_visit_charge, "next time")

        old_charge = charge
        old_pos = pos

    def search_neighbor_field_with_way(pos):
        # return: (neighbor position, whether position is used as way)
        neighbor_pos = None
        for neighbor in get_free_neighbor_fields(pos):
            if (f := field[neighbor]) == 1:
                return neighbor, True
            elif f == 0:
                neighbor_pos = neighbor
        return neighbor_pos, False

    # search a way to get the last battery empty
    if charge <= 2 or pos == batteries[0][0]:  # batteries[0][0] is robot battery pos
        # only need one neighbor field
        neighbor = search_neighbor_field_with_way(pos)[0]
        if neighbor is not None:
            field[neighbor] = 1
        else:
            # need to remove last battery
            del batteries[-1]
    else:
        neighbor_field1 = None
        neighbor_field2 = None
        found_one_field_with_way = False
        for neighbor in get_free_neighbor_fields(pos):
            if (f := field[neighbor]) == 1:
                neighbor_neighbor, is_way = search_neighbor_field_with_way(neighbor)
                neighbor_field1 = neighbor
                neighbor_field2 = neighbor_neighbor
                if is_way:
                    break
                found_one_field_with_way = True
            elif f == 0:
                if found_one_field_with_way:
                    continue
                neighbor_neighbor, is_way = search_neighbor_field_with_way(neighbor)
                neighbor_field1 = neighbor
                neighbor_field2 = neighbor_neighbor
                if is_way:
                    found_one_field_with_way = True
        if neighbor_field1 is not None and neighbor_field2 is not None:
            field[neighbor_field1] = 1
            field[neighbor_field2] = 1
        else:
            del batteries[-1]

    print("\n\nremoving battery charge from batteries that could not be visited multiple times")
    for pos, (multi_visit_charge, battery_num_from_which_charge_originates) in multi_visit_fields.items():
        print("battery on pos", pos, "could not be visited multiple times")
        pos, old_charge = batteries[battery_num_from_which_charge_originates]
        new_charge = old_charge - multi_visit_charge
        print("  change charge on field", pos, "from", old_charge, "to", new_charge)
        batteries[battery_num_from_which_charge_originates] = (pos, new_charge)

    for start_x in range(size):
        if not np.all(field[:, start_x] == 0):
            break

    for start_y in range(size):
        if not np.all(field[start_y, :] == 0):
            break

    for end_x in range(size-1, -1, -1):
        if not np.all(field[:, end_x] == 0):
            break

    for end_y in range(size-1, -1, -1):
        if not np.all(field[end_y, :] == 0):
            break

    if EXTENDED_FILE_FORMAT:
        shift = Vector2(start_x, start_y)
        field_size = (end_x - start_x + 1, end_y - start_y + 1)
        print("\n\nnot multi-visited", [pos - shift for pos in multi_visit_fields])
        print("all multi-visited fields", [pos - shift for pos in all_multi_visit])

        def battery_to_str(pos, charge):
            shifted = pos - shift
            return (str(shifted.x) + "," + str(shifted.y) + "," + str(charge) + ":" + str(pos.x) + "," +
                    str(pos.y) + "\n")

        def pos_to_str(pos):
            shifted = pos - shift
            return str(shifted.x) + "," + str(shifted.y)

        return (
            str(field_size[0]) + "," + str(field_size[1]) + "=0\n", battery_to_str(*batteries[0]),
            "*", "\n", *(battery_to_str(pos, charge) for pos, charge in batteries[1:]),
            "-\nLABELS\n", *((way_pos-shift).compact_str() + ":" +
                             " ".join(pos.compact_str() for pos in positions) + "\n"
                             for way_pos, positions in labels.items()),
            "-\n#!solution=", ";".join(pos_to_str(pos) for pos in solution),
        )
    else:
        shift = Vector2(start_x-1, start_y-1)  # origin is (1, 1), not (0, 0)
        size = max(end_x - start_x, end_y - start_y) + 1

        def battery_to_str(pos, charge):
            pos = pos - shift
            return str(pos.x) + "," + str(pos.y) + "," + str(charge) + "\n"

        return (str(size), "\n", battery_to_str(*batteries[0]),
                str(len(batteries)-1), "\n", *(battery_to_str(pos, charge) for pos, charge in batteries[1:]))


if __name__ == "__main__":
    directory = "generated_examples"
    for i in range(10):
        filename = f"example{i}.txt"
        print("\n############")
        print(filename)
        rand = Random(i)
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            f.writelines(generate_stromrallye(FIELD_SIZE, BATTERY_COUNT, rand, NEW_CHARGE,
                                              PROBABILITY_MARK_AS_MULTI_VISIT, PROBABILITY_DO_MULTI_VISIT,
                                              PROBABILITY_TAKE_FASTEST))

        way_fields = None
        robot = aufgabe1a.Robot.from_file(path)
        robot.create_fake_batteries()

        graph = robot.save_battery_field_graph(name=filename + "_fields", directory=directory, format="svg")
        graph.render()
        graph = robot.save_graph(name=filename, directory=directory, format="svg")
        graph.render()

        way = robot.find_way(do_print=False)
        aufgabe1a.output_way(way, robot)
        if way:
            way_fields = list(robot.generate_full_way(way))
        else:
            way_fields = None

        with open(path, "r") as f:
            drawing = save_svg(f, draw_labels=False,
                               solution=way_fields,
                               draw_solution_from_file=True,
                               use_positions_instead_of_labels=True)
        drawing.filename = path + ".solution.svg"
        drawing.save(True)

        with open(path, "r") as f:
            drawing = save_svg(f, draw_labels=False,
                               draw_solution_from_file=False,
                               use_positions_instead_of_labels=True)
        drawing.filename = path + ".svg"
        drawing.save(True)
