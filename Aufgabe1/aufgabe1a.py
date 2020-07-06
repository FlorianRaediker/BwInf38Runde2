"""
Aufgabe 1: Stromrallye (a)

Dieses Skript ist Teil der Einsendung für die

    2. Runde des
    38. Bundeswettbewerbs Informatik

von

    Florian Rädiker.

Teilnahme-ID: 52570
"""
import argparse
import os
import time
import itertools
import math
import operator
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Set, Tuple, Dict, Union

import graphviz
import numpy as np

from util import Vector2, StromrallyeParser, save_svg


class BaseField:
    id: int
    pos: Vector2
    _connections: List["BatteryFieldConnection"]
    _tp_connections: List["BatteryFieldConnection"]  # connections to tp fields

    def __init__(self, pos):
        self.pos = pos
        self.id = -1
        self._fields2connections = {}
        self._connections = []
        self._tp_connections = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pos})"

    @property
    def connections(self):
        return self._connections

    @property
    def tp_connections(self):
        return self._tp_connections

    def add_connection(self, connection):
        raise NotImplementedError


class BatteryField(BaseField):
    initial_battery: "Battery"
    active_battery: "Battery"
    free_fields_around: List[Union[Vector2, "TeleportationField"]]
    small_way_lengths_without_longer_ways: Set[int]
    neighbors: Set["BatteryField"]
    is_visited: bool
    batteries: Dict[int, "Battery"]

    def __init__(self, pos: Vector2, initial_battery_charge):
        super().__init__(pos)
        self.initial_battery = self.active_battery = Battery(initial_battery_charge, self)
        self.batteries = {}
        self.free_fields_around = []
        self.small_way_lengths_without_longer_ways = set()
        self.neighbors = set()
        self.is_visited = False

    def __hash__(self):
        return self.pos.__hash__() + 7

    def add_connection(self, connection):
        if type(connection.other_field) == TeleportationField:
            self.tp_connections.append(connection)
        else:
            self._connections.append(connection)

    def get_or_create_fake_battery(self, charge):
        """
        :param charge: the charge the new battery should have
        :return: (whether battery is new, the battery)
        """
        if charge in self.batteries:
            return False, self.batteries[charge]
        if charge == self.initial_battery.charge:
            self.batteries[charge] = (battery := self.initial_battery)
        else:
            self.batteries[charge] = (battery := Battery(charge, self))
        return True, battery


class TeleportationField(BaseField):
    paired_field: "TeleportationField"  # the TeleportationField to which this teleportation field is connected

    def __init__(self, pos, paired_field=None):
        super().__init__(pos)
        self.paired_field = paired_field

    def add_connection(self, connection):
        if type(connection.other_field) == TeleportationField:
            self._tp_connections.append(connection)
        else:
            self.connections.append(connection)


@dataclass
class Battery:
    charge: int
    battery_field: "BatteryField"
    connections: List["BatteryConnection"] = field(default_factory=list)
    id: int = -1
    predecessor_count: int = 0
    reachable_fields: Optional[List[Set[int]]] = None

    def __hash__(self):
        return self.battery_field.__hash__() + self.charge

    def __repr__(self):
        return f"Battery({self.charge}, {self.battery_field.pos})"


@dataclass
class BatteryFieldConnection:
    other_field: Union[BatteryField, TeleportationField]
    distance: int
    way: Union[List[Union[Vector2, str]], Tuple[list, list, list]]
    distance_greater_than_two: int = -1
    longer_way: Optional[List[Union[Vector2, str]]] = None


@dataclass
class BatteryConnection:
    new_battery: Battery  # the battery that will replace the current battery in field new_battery.battery_field
    # the following two attributes are only used for generating the full way (every position) in Robot.generate_full_way
    field_connection: BatteryFieldConnection  # the battery field connection that is used to create this connection
    uses_long_field_connection: bool  # whether field_connection.way or field_connection.longer_way is used

    def __repr__(self):
        return f"BatteryConnection({self.new_battery.battery_field.pos}, {self.new_battery.charge})"


class Robot:
    distances: np.ndarray
    battery_fields: List[BatteryField]
    robot_battery_field: BatteryField
    battery_count: Optional[int]

    def __init__(self, battery_fields: List[BatteryField], robot_battery_field: BatteryField, size: Tuple[int, int]):
        self.battery_fields = battery_fields
        self.robot_battery_field = robot_battery_field
        self.battery_count = None
        self.size = size

    def save_graph(self, scale=2.8, node_distance=1,
                   way: Optional[Sequence[Tuple[Battery, Optional[BatteryConnection]]]] = None,
                   **graph_kwargs):
        @lru_cache(maxsize=None)
        def get_battery_node_name(battery: Battery):
            return f"battery{battery.battery_field.pos.x}_{battery.battery_field.pos.y}_{battery.charge}"

        graph = graphviz.Digraph(**graph_kwargs)
        graph.engine = "neato"
        graph.attr(compound="true")
        battery_field_names = {}
        for battery_field in self.battery_fields:
            subgraph_name = f"{battery_field.pos.x}_{battery_field.pos.y}"
            name = "cluster" + subgraph_name
            batteries = sorted(battery_field.batteries.values(), key=operator.attrgetter("charge"))
            initial_battery = battery_field.initial_battery
            initial_battery_node_name = get_battery_node_name(initial_battery)
            battery_field_names[battery_field.pos] = (name, initial_battery_node_name)
            with graph.subgraph(name=name) as sg:
                sg.attr(label=f"{battery_field.pos.x},{battery_field.pos.y}")
                i = 0
                width = int(math.ceil(math.sqrt(len(battery_field.batteries.values()))))
                for battery in batteries:
                    if battery == initial_battery:
                        sg.node(get_battery_node_name(battery), str(battery.charge) + "," + str(battery.id),
                                pos=f"{battery_field.pos.x * scale + (i % width) * node_distance},"
                                    f"{(self.size[1] - battery_field.pos.y) * scale + - (i // width * node_distance)}" +
                                    "!",
                                fillcolor="yellow" if self.robot_battery_field != battery_field else "green",
                                style="filled")
                    else:
                        sg.node(get_battery_node_name(battery), str(battery.charge) + "," + str(battery.id),
                                pos=f"{battery_field.pos.x * scale + (i % width) * node_distance},"
                                    f"{(self.size[1] - battery_field.pos.y) * scale + - (i // width * node_distance)}" +
                                    "!")
                    i += 1
        for battery_field in self.battery_fields:
            for battery in battery_field.batteries.values():
                battery_name = get_battery_node_name(battery)
                for battery_connection in battery.connections:
                    other_battery = battery_connection.new_battery
                    if way is not None and (t := (battery, battery_connection)) in way:
                        graph.edge(battery_name, get_battery_node_name(other_battery), label=str(way.index(t)*2+1),
                                   fillcolor="red", color="red")
                    else:
                        graph.edge(battery_name, get_battery_node_name(other_battery),
                                   fillcolor="none", color="#00000055")

        if way is not None:
            def add_edge(from_, to, num):
                graph.edge(get_battery_node_name(from_), get_battery_node_name(to),
                           fillcolor="orange", color="orange", label=str(num))

            add_edge(self.robot_battery_field.batteries[0], way[0][0], 0)

            if len(way) > 1:
                old_battery = way[0][1].new_battery

                for i, (battery, connection) in enumerate(way[1:], 1):
                    add_edge(old_battery, battery, i*2)
                    if connection is not None:  # last connection is None
                        old_battery = connection.new_battery

        return graph

    def save_battery_field_graph(self, scale=2.8, **graph_kwargs):
        graph = graphviz.Digraph(**graph_kwargs)
        graph.engine = "neato"
        graph.attr(compound="true")
        battery_field_names = {}
        for battery_field in self.battery_fields:
            name = f"{battery_field.pos.x}_{battery_field.pos.y}"
            free = (';'.join(p.compact_str() if type(p) == Vector2
                             else (p.pos.compact_str() if p else "None")  # p is TeleportationField
                             for p in battery_field.free_fields_around)
                    if battery_field.free_fields_around else 'none')
            swlolw = (battery_field.small_way_lengths_without_longer_ways
                      if battery_field.small_way_lengths_without_longer_ways else "none")
            label = f"{battery_field.pos.x},{battery_field.pos.y}\nic: {battery_field.initial_battery.charge}\n" \
                    f"free: {free}\nswlolw: {swlolw}"
            battery_field_names[battery_field] = name
            if self.robot_battery_field == battery_field:
                graph.node(name, label,
                           pos=f"{battery_field.pos.x * scale},{(self.size[1] - battery_field.pos.y) * scale}!",
                           fillcolor="green", style="filled")
            else:
                graph.node(name, label,
                           pos=f"{battery_field.pos.x * scale},{(self.size[1] - battery_field.pos.y) * scale}!")

        for battery_field in self.battery_fields:
            for connection in battery_field.connections:
                graph.edge(battery_field_names[battery_field], battery_field_names[connection.other_field],
                           label=str(connection.distance) + "  (" + str(connection.distance_greater_than_two) + ") ")
        return graph

    @staticmethod
    def from_file(path: str):
        with open(path, "r", encoding="utf-8") as f:
            parser = StromrallyeParser(f)
            size = parser.size()  # (width, height)
            pos, charge, _ = parser.robot_battery()
            robot_battery_field = BatteryField(pos, charge)
            battery_fields = []
            tp_fields = []
            for type_, info in parser.batteries():
                if type_ == "battery":
                    pos, charge, _ = info
                    battery_fields.append(BatteryField(pos, charge))
                else:
                    assert type_ == "tp_field"
                    pos1, pos2, _ = info
                    tp1 = TeleportationField(pos1)
                    tp2 = TeleportationField(pos2, tp1)
                    tp1.paired_field = tp2
                    tp_fields.append(tp1)
                    tp_fields.append(tp2)
            parser.end()
        return Robot.from_battery_fields(size, robot_battery_field, battery_fields, tp_fields)

    @staticmethod
    def from_battery_fields(size: Tuple[int, int], robot_battery_field: BatteryField,
                            original_battery_fields: List[BatteryField], tp_fields: List[TeleportationField]):
        """
        Creates a new Robot.
        :param size: size of game board
        :param robot_battery_field: the battery field which represents the robot (NOT included in param battery_fields)
        :param original_battery_fields: list of all other battery fields
        :param tp_fields: list of all teleportation fields
        :return: Robot instance
        """
        def create_new_connections(battery_field, other_battery_field, distance, way):
            # connections contains the two connections between the two fields or None (if the second battery
            # field is robot battery field), like this:
            # [connection from battery_field to other_battery_field,
            #  connection from other_battery_field to battery_field]
            connections = []
            if other_battery_field != robot_battery_field:
                connection = BatteryFieldConnection(other_battery_field, distance, way)
                # Attribute 'longer_way' und 'distance_greater_than_two' werden später gesetzt (wenn es
                # überhaupt einen längeren Weg gibt). 'distance_greater_than_two' ist im Moment -1.
                connections.append(connection)
                battery_field.add_connection(connection)
            else:
                connections.append(None)
            if battery_field != robot_battery_field:
                connection = BatteryFieldConnection(battery_field, distance, way[::-1])
                connections.append(connection)
                other_battery_field.add_connection(connection)
            else:
                connections.append(None)
            return connections

        def improve_connections(pos, distance_greater_than_two, longer_way):
            # set distance_greater_than_two and longer_way for connections on position 'pos'
            # diese Funktion wird aufgerufen, wenn ein längerer Weg als 2 für pos gefunden wurde.
            # Die Verbindungen aus connections_with_small_distances (s.u.) werden aktualisiert.
            connections = connections_with_small_distances[pos]
            if (c := connections[0]) is not None:
                c.longer_way = longer_way
                c.distance_greater_than_two = distance_greater_than_two
            if (c := connections[1]) is not None:
                c.longer_way = longer_way[::-1]
                c.distance_greater_than_two = distance_greater_than_two
            connections_with_small_distances[pos] = None

        original_battery_fields.append(robot_battery_field)
        battery_fields = original_battery_fields.copy()
        battery_fields.extend(tp_fields)
        all_battery_fields = battery_fields.copy()  # battery fields, sorted by id
        battery_fields.sort(key=lambda b: (b.pos.y, b.pos.x))
        id_ = 0
        for id_, battery_field in enumerate(original_battery_fields, id_):
            battery_field.id = id_
        robot_battery_field.id = id_
        id_ += 1
        for id_, teleportation_field in enumerate(tp_fields, id_):
            teleportation_field.id = id_

        field = np.full((size[1], size[0]), -1, dtype=np.int_)  # contains battery field ids
        for battery_field in battery_fields:
            if battery_field != robot_battery_field:
                if (field[battery_field.pos] != -1 or
                        battery_field.pos == robot_battery_field.pos):  # check that battery field is not located on
                                                                        # robot battery field
                    raise ValueError(f"There are at least two batteries or teleportation fields at position "
                                     f"{battery_field.pos}")
                field[battery_field.pos] = battery_field.id

        battery_count = len(battery_fields)
        for battery_num in range(battery_count - 1):  # no need to include last battery
            battery_field = battery_fields[battery_num]
            print("searching ways for field", battery_num, battery_field.pos)
            if type(battery_field) == BatteryField and battery_field.initial_battery.charge == 0:
                # if a battery field has charge 0 from the beginning, no connections need to be searched
                continue
            batteries_for_maze_routing = np.full(field.shape, -1, dtype=np.int_)  # contains battery ids for maze
                                                                                  # routing
            connections_with_small_distances = np.empty(field.shape, dtype=list)  # contains connections with
                                                                                  # distance <= 2 for maze routing
            batteries_for_maze_routing_count = 0  # count of batteries for maze routing
            batteries_with_small_distance_count = 0  # count of batteries to which a greater distance should be found
                                                     # using maze routing
            for other_battery_num in range(battery_num + 1, battery_count):
                assert battery_num != other_battery_num
                other_battery_field = battery_fields[other_battery_num]
                if type(other_battery_field) == BatteryField and other_battery_field.initial_battery.charge == 0:
                    # see comment above
                    continue
                assert battery_field != other_battery_field
                min_x, max_x = sorted((battery_field.pos.x, other_battery_field.pos.x))
                # battery_field.pos.y must be <= other_battery_field.pos.y, because battery fields were sorted
                horizontal = field[battery_field.pos.y, min_x+1:max_x]
                vertical = field[battery_field.pos.y+1:other_battery_field.pos.y, other_battery_field.pos.x]
                if battery_field.pos.y != other_battery_field.pos.y and min_x != max_x:
                    corner = field[battery_field.pos.y, other_battery_field.pos.x]
                else:
                    corner = -1
                if np.all(horizontal == -1) and np.all(vertical == -1) and corner == -1:
                    distance = (max_x - min_x) + (other_battery_field.pos.y - battery_field.pos.y)
                    # generate way from battery_field to other_battery_field
                    way = [Vector2(x, battery_field.pos.y) for x in
                           (range(battery_field.pos.x+1, other_battery_field.pos.x)
                            if battery_field.pos.x <= other_battery_field.pos.x else
                            range(battery_field.pos.x-1, other_battery_field.pos.x, -1))]  # horizontal
                    if battery_field.pos.y != other_battery_field.pos.y and min_x != max_x:
                        way.append(Vector2(other_battery_field.pos.x, battery_field.pos.y))  # corner
                    way.extend(Vector2(other_battery_field.pos.x, y) for y in
                               range(battery_field.pos.y+1, other_battery_field.pos.y))  # vertical

                    connections = create_new_connections(battery_field, other_battery_field, distance, way)
                    if distance > 2:
                        for connection in connections:
                            if connection:
                                connection.distance_greater_than_two = distance
                    else:
                        # save these connections for maze routing to find a greater distance
                        batteries_with_small_distance_count += 1
                        connections_with_small_distances[other_battery_field.pos] = connections
                else:
                    # no easy way was found to other_battery_field, so a way must be searched with maze routing
                    batteries_for_maze_routing[other_battery_field.pos] = other_battery_num
                    batteries_for_maze_routing_count += 1

            if batteries_for_maze_routing_count != 0 or batteries_with_small_distance_count != 0:
                # not all ways could be easily detected, so use maze routing
                # array 'visited' contains sets with all distances with which a field has been visited
                visited = np.frompyfunc(set, 0, 1)(np.empty(field.shape, dtype=object))
                one_visit_positions = {battery_field.pos: []}  # {<position>: <way to this position>}
                double_visit_positions = {}
                for distance in itertools.count(1):  # start with distance 1, all newly found fields have this distance
                    new_one_visit_positions = {}
                    new_double_visit_positions = {}
                    do_break = False
                    for pos, way in one_visit_positions.items():
                        visited[pos].add(distance)

                        for new_pos in pos.neighbor_fields(0, field.shape[1]-1, 0, field.shape[0]-1):
                            assert 0 <= new_pos.x < field.shape[1] and 0 <= new_pos.y < field.shape[0]
                            if connections_with_small_distances[new_pos] is not None:
                                # this field contains a battery for which a greater distance should be found
                                if distance > 2:
                                    improve_connections(new_pos, distance, way)
                                    batteries_with_small_distance_count -= 1
                                    if batteries_for_maze_routing_count == 0 == batteries_with_small_distance_count:
                                        # there are no fields left for maze routing
                                        do_break = True
                                        break
                            if not visited[new_pos]:
                                num = batteries_for_maze_routing[new_pos]
                                if num != -1:
                                    # a battery is on this field
                                    other_battery_field = battery_fields[num]
                                    assert battery_field != other_battery_field
                                    batteries_for_maze_routing_count -= 1
                                    batteries_for_maze_routing[new_pos] = -1
                                    connections = create_new_connections(battery_field, other_battery_field, distance,
                                                                         way)
                                    if distance > 2:
                                        for connection in connections:
                                            if connection:
                                                connection.distance_greater_than_two = distance
                                    else:
                                        batteries_with_small_distance_count += 1
                                        connections_with_small_distances[other_battery_field.pos] = connections

                                    if batteries_for_maze_routing_count == 0 == batteries_with_small_distance_count:
                                        # there are no fields left for maze routing
                                        do_break = True
                                        break
                                if field[new_pos] == -1:
                                    new_one_visit_positions[new_pos] = way + [new_pos]
                            else:
                                # this field has already been visited, so this is a new position for
                                # other_positions
                                if field[new_pos] == -1:
                                    # the field must be empty
                                    new_double_visit_positions[new_pos] = way + [new_pos]
                        if do_break:
                            break
                    if do_break:
                        # instead of using do_break, for-else with continue could be used, but using an extra variable
                        # is easier to understand
                        break
                    if batteries_with_small_distance_count != 0 and distance < 5:
                        # there are still battery fields left for which greater distances are searched
                        for pos, way in double_visit_positions.items():
                            visited[pos].add(distance)
                            for new_pos in pos.neighbor_fields(0, field.shape[1]-1, 0, field.shape[0]-1):
                                if distance not in visited[new_pos]:
                                    if connections_with_small_distances[new_pos]:
                                        # this field contains a battery for which a greater distance should be found
                                        if distance > 2:
                                            improve_connections(new_pos, distance, way)
                                            batteries_with_small_distance_count -= 1
                                            if batteries_for_maze_routing_count == 0 == \
                                                    batteries_with_small_distance_count:
                                                # there are no fields left for maze routing
                                                do_break = True
                                                break
                                    if field[new_pos] == -1:
                                        new_double_visit_positions[new_pos] = way + [new_pos]
                            if do_break:
                                break
                    if do_break or (not new_one_visit_positions and (distance > 5 or not new_double_visit_positions)):
                        break
                    one_visit_positions = new_one_visit_positions
                    double_visit_positions = new_double_visit_positions

        def get_teleportation_reachable_battery_fields(start_field: BatteryField):
            even_distances = {battery_or_tp_field: (math.inf, None) for battery_or_tp_field in battery_fields}
            odd_distances = even_distances.copy()
            del even_distances[start_field]
            for connection in start_field.tp_connections:
                distance_to_neighbor = connection.distance + 1  # +1 because teleportation takes 1 step
                neighbor = connection.other_field.paired_field
                way_to_neighbor = connection.way + [connection.other_field.pos, "tp"]
                if distance_to_neighbor % 2 == 0:
                    if neighbor in even_distances and distance_to_neighbor < even_distances[neighbor][0]:
                        even_distances[neighbor] = (distance_to_neighbor, way_to_neighbor)
                else:
                    if neighbor in odd_distances and distance_to_neighbor < odd_distances[neighbor][0]:
                        odd_distances[neighbor] = (distance_to_neighbor, way_to_neighbor)
            while even_distances or odd_distances:
                if even_distances:
                    current_even_node, (even_distance, even_way) = min(even_distances.items(), key=lambda n: n[1][0])
                else:
                    even_distance = math.inf
                if odd_distances:
                    current_odd_node, (odd_distance, odd_way) = min(odd_distances.items(), key=lambda n: n[1][0])
                else:
                    odd_distance = math.inf
                if even_distance == math.inf == odd_distance:
                    break
                if even_distance < odd_distance:
                    # smallest even distance is smaller than smallest odd distance
                    current_node = current_even_node
                    distance = even_distance
                    way = even_way
                    nodes = even_distances
                else:
                    current_node = current_odd_node
                    distance = odd_distance
                    way = odd_way
                    nodes = odd_distances
                del nodes[current_node]
                if way and type(current_node) == BatteryField:
                    yield current_node, way, distance
                else:
                    for connection in itertools.chain(current_node.connections, current_node.tp_connections):
                        distance_to_neighbor = distance + connection.distance
                        neighbor = connection.other_field
                        way_to_neighbor = way + [current_node.pos] + connection.way
                        if distance_to_neighbor % 2 == 0:
                            if neighbor in even_distances and distance_to_neighbor < even_distances[neighbor][0]:
                                even_distances[neighbor] = (distance_to_neighbor, way_to_neighbor)
                        else:
                            if neighbor in odd_distances and distance_to_neighbor < odd_distances[neighbor][0]:
                                odd_distances[neighbor] = (distance_to_neighbor, way_to_neighbor)

        def get_free_neighbor_fields(pos: Vector2):
            # yield: position (Vector2) or teleportation field (TeleportationField)
            for p in pos.neighbor_fields(0, field.shape[1] - 1, 0, field.shape[0] - 1):
                if field[p] == -1:
                    yield p
                elif type(f := all_battery_fields[field[p]]) == TeleportationField:
                    yield f

        for tp_field in tp_fields:
            tp_field.add_connection(BatteryFieldConnection(tp_field.paired_field, 1, ["tp"]))

        for battery_field in original_battery_fields:
            # calculate small way lengths without longer ways ("swlolw", siehe Dokumentation)
            for connection in battery_field.connections:
                if connection.distance < 3 and connection.distance_greater_than_two not in (3, 4):
                    battery_field.small_way_lengths_without_longer_ways.add(connection.distance)

            # search free neighbor fields (could be done with maze routing)
            for free_pos1 in get_free_neighbor_fields(battery_field.pos):
                if type(free_pos1) == TeleportationField:
                    # teleportation field is neighbor field, there are two free fields around
                    battery_field.free_fields_around = [free_pos1, None]
                    break
                battery_field.free_fields_around = [free_pos1]  # at least one free field exists
                for free_pos2 in get_free_neighbor_fields(free_pos1):
                    if type(free_pos2) == TeleportationField:
                        # teleportation field is neighbor of (free) neighbor field
                        battery_field.free_fields_around = [free_pos1, free_pos2]
                        break
                    # two connected free fields were found
                    # note that free_pos1 can be == robot_battery_field.pos
                    #   (even if battery_field == robot_battery_field)
                    battery_field.free_fields_around = [free_pos1, free_pos2]
                    break
                else:
                    # loop did not break
                    continue
                break

            # add connection from battery_field to battery_field
            if battery_field.initial_battery.charge != 0 and battery_field != robot_battery_field:
                if len(battery_field.free_fields_around) == 2:
                    battery_field.add_connection(BatteryFieldConnection(battery_field,
                                                                        2, [battery_field.free_fields_around[0]],
                                                                        4, [battery_field.free_fields_around[i]
                                                                            for i in (0, 1, 0)]))
                elif len(battery_field.free_fields_around) == 1:
                    # only possible length is 2, 4 or more is not possible
                    battery_field.add_connection(BatteryFieldConnection(battery_field,
                                                                        2, [battery_field.free_fields_around[0]]))

            # add connections which use teleportation fields
            for other_battery_field, way, distance in get_teleportation_reachable_battery_fields(battery_field):
                # convert way to some way that generate_full_way "understands" -> convert to tuple:
                # (first way, way of two fields that can be repeated, rest of way)
                tp_index = way.index("tp")
                first_tp_field_pos = way[tp_index-1]
                second_tp_field_pos = way[tp_index+1]
                way = (way[:tp_index+1],
                       [second_tp_field_pos, "tp", first_tp_field_pos, "tp"],
                       way[tp_index+1:])
                battery_field.add_connection(BatteryFieldConnection(other_battery_field, distance, way,
                                                                    distance, None))
            # cleanup: remove redundant battery connections
            battery_field.connections.sort(key=lambda b: b.distance)
            for i in range(len(battery_field.connections)-1, -1, -1):
                connection = battery_field.connections[i]
                for other_connection in battery_field.connections[:i]:
                    if other_connection.other_field == connection.other_field:
                        assert other_connection != connection
                        assert other_connection.distance <= connection.distance
                        if (connection.distance % 2 == other_connection.distance % 2 and
                                ((connection.distance_greater_than_two >= other_connection.distance != -1) or
                                 connection.distance_greater_than_two == -1 == other_connection.distance)):
                            del battery_field.connections[i]
                            break

        return Robot(original_battery_fields, robot_battery_field, size)

    def create_fake_batteries(self):
        battery_id = 0  # counter for battery ids
        #for field in self.battery_fields:  # connections have already been sorted by from_battery_fields
        #    field.connections.sort(key=lambda c: c.distance)

        def search_connections(current_battery: Battery):
            def add_neighbor(charge, field_connection: BatteryFieldConnection, uses_long_way):
                # add a new neighbor battery with charge 'charge' on field 'field_connection.other_field'
                nonlocal battery_id
                if charge == 0:
                    current_battery.connections.append(
                        BatteryConnection(field_connection.other_field.get_or_create_fake_battery(0)[1],
                                          field_connection,
                                          uses_long_way))
                else:
                    is_new, new_battery = field_connection.other_field.get_or_create_fake_battery(charge)
                    if is_new:
                        new_battery.id = battery_id
                        battery_id += 1
                        # the battery is new, so search for neighbors
                        search_connections(new_battery)
                    current_battery.connections.append(
                        BatteryConnection(new_battery, field_connection, uses_long_way))

            for field_connection in current_battery.battery_field.connections:
                if field_connection.distance > current_battery.charge:
                    # battery connections are sorted (see beginning of create_fake_batteries), so all following
                    # distances are too large as well
                    break
                if field_connection.distance == current_battery.charge:
                    add_neighbor(0, field_connection, False)
                else:
                    assert field_connection.distance < current_battery.charge
                    if (field_connection.distance % 2 == current_battery.charge % 2 and
                            (current_battery.charge < 3 or
                             # if current_battery.charge > 2, the way longer than 2 must have a distance smaller than
                             # or equal to current_battery.charge
                             (field_connection.distance_greater_than_two != -1 and
                              field_connection.distance_greater_than_two <= current_battery.charge))
                    ):
                        # a battery with charge 0 can be created on battery_connection.other_field
                        # batteries with charge 0 have no id (id is -1)
                        add_neighbor(0, field_connection,
                                     current_battery.charge > 2 and field_connection.distance < 3)
                    # the charge that a battery in field battery_connection.other_field can get at most
                    maximum_remaining_charge = current_battery.charge - field_connection.distance
                    # create a new fake battery on field battery_connection.other_field (if it doesn't exist already)
                    add_neighbor(maximum_remaining_charge, field_connection, False)
                    # additionally, add fake batteries for every "small way length without longer way"
                    for small_distance in field_connection.other_field.small_way_lengths_without_longer_ways:
                        if small_distance != maximum_remaining_charge:
                            if maximum_remaining_charge % 2 == small_distance % 2:
                                distance = current_battery.charge - small_distance
                                if (distance < 3 or (field_connection.distance_greater_than_two != -1 and
                                                     field_connection.distance_greater_than_two <= distance)):
                                    add_neighbor(small_distance, field_connection,
                                                 distance > 2 and field_connection.distance < 3)

        for battery_field in self.battery_fields:
            is_new, battery = battery_field.get_or_create_fake_battery(battery_field.initial_battery.charge)
            # The initial battery could also be a fake battery. In that case, is_new can be False.
            if is_new:
                battery.id = battery_id
                battery_id += 1
                search_connections(battery)

        # sort all battery connections
        for battery_field in self.battery_fields:
            for battery in battery_field.batteries.values():
                battery.connections.sort(
                    key=lambda c: (
                        # batteries which are located on another battery field first
                        0 if c.new_battery.battery_field != battery.battery_field else 1,
                        # charged batteries which have no connections last
                        0 if c.new_battery.connections or c.new_battery.charge == 0 else 1,
                        # batteries with higher charge first
                        -c.new_battery.charge,
                        # batteries on more distant fields first
                        -c.field_connection.distance
                    )
                )
                battery_field.neighbors.update(c.new_battery.battery_field for c in battery.connections)

        if 0 not in self.robot_battery_field.batteries:
            # self.robot_battery_field.get_active_battery().charge is odd,
            # therefore no battery with charge 0 has been created, but robot battery field needs an empty battery
            self.robot_battery_field.get_or_create_fake_battery(0)

        self.battery_count = battery_id  # only used for outputting how many vertices the graph has

    def find_way(self, do_print=True) -> Optional[List[Tuple[Battery, Optional[BatteryConnection]]]]:
        def is_graph_connected(current_battery: Battery):
            fields = deque(c.new_battery.battery_field for c in current_battery.connections)
            unvisited_fields = active_battery_fields
            if unvisited_fields == 0:
                return True
            while fields:
                field = fields.pop()
                if unvisited_fields & (mask := 1 << field.id):
                    unvisited_fields &= ~mask
                    if unvisited_fields == 0:
                        return True
                    fields.extendleft(field.neighbors)
            return False

        cache = set()
        cache_hits = 0

        def find_way_backtracking(current_battery: Battery, new_battery: Battery, indent=0):
            nonlocal active_battery_fields, batteries, cache_hits
            if do_print and indent < 400:
                print(indent * "  ", "trying ", current_battery.battery_field.pos, " ", current_battery.charge, " ",
                      new_battery.charge, sep="")
            # mark the current battery as inactive
            batteries &= ~(1 << current_battery.id)

            if new_battery.charge != 0:
                # mark battery with id new_battery.id as active (set bit to 1)
                batteries |= 1 << new_battery.id
            else:
                # mark battery field with id current_battery.battery_field.id as inactive (set bit to 0)
                active_battery_fields &= ~(1 << current_battery.battery_field.id)

            key = (batteries, current_battery)
            if key in cache:
                cache_hits += 1
                active_battery_fields |= 1 << current_battery.battery_field.id
                if new_battery.charge != 0:
                    batteries &= ~(1 << new_battery.id)
                batteries |= 1 << current_battery.id
                return None
            cache.add(key)

            if batteries == 0:
                # check that current_battery.battery_fields has enough free space
                if (current_battery.charge <= 1 or
                        (current_battery.charge == 2 and len(current_battery.battery_field.free_fields_around) >= 1) or
                        len(current_battery.battery_field.free_fields_around) == 2):
                    # all batteries are inactive, so a way has been found
                    return [(current_battery, None)]
                return None

            # check that the graph is still connected
            if not is_graph_connected(current_battery):
                if do_print and indent < 400:
                    print(indent * "  ", "disconnected", sep="")
                active_battery_fields |= 1 << current_battery.battery_field.id
                if new_battery.charge != 0:
                    batteries &= ~(1 << new_battery.id)
                batteries |= 1 << current_battery.id
                return None

            # BatteryField.active_battery stores the battery currently lying on the battery field
            current_battery.battery_field.active_battery = new_battery
            old_is_visited = current_battery.battery_field.is_visited
            current_battery.battery_field.is_visited = True

            # split current_battery.connections according to whether other_field is visited or unvisited
            unvisited = []
            visited = []
            for connection in current_battery.connections:
                if connection.new_battery.battery_field.is_visited:
                    visited.append(connection)
                else:
                    unvisited.append(connection)
            for connection in itertools.chain(unvisited, visited):
                other_field = connection.new_battery.battery_field
                if other_field.active_battery.charge != 0:
                    way = find_way_backtracking(other_field.active_battery, connection.new_battery, indent+1)
                    if way is not None:
                        return [(current_battery, connection)] + way

            # reverse changes made to active batteries and whether current_battery.battery_field is visited
            current_battery.battery_field.active_battery = current_battery
            current_battery.battery_field.is_visited = old_is_visited
            active_battery_fields |= 1 << current_battery.battery_field.id
            if new_battery.charge != 0:
                batteries &= ~(1 << new_battery.id)
            batteries |= 1 << current_battery.id
            return None

        robot_battery = self.robot_battery_field.active_battery
        if robot_battery.charge == 0:
            return None
        active_battery_fields = 0
        batteries = 0
        for field in self.battery_fields:
            if field.initial_battery.charge != 0:
                active_battery_fields |= 1 << field.id
                batteries |= 1 << field.initial_battery.id

        print("field count:", len(self.battery_fields), "battery count:", self.battery_count)
        res = find_way_backtracking(robot_battery, self.robot_battery_field.batteries[0])
        print("cache size:", len(cache), "hits:", cache_hits)
        return res

    def generate_full_way(self, way: List[Tuple[Battery, Optional[BatteryConnection]]]):
        for battery, connection in way:
            yield battery.battery_field.pos
            if connection is not None:
                way = (connection.field_connection.longer_way
                       if connection.uses_long_field_connection else
                       connection.field_connection.way)
                if type(way) != tuple:
                    # convert way (list) to tuple
                    way = (way, way[-2:], [])
                yield from way[0]  # yield first way segment
                distance = battery.charge - connection.new_battery.charge  # charge by which the battery needs to be
                                                                           # discharged
                way_distance = len(way[0]) - way[0].count("tp") + len(way[2]) - way[2].count("tp") + 1
                diff = distance - way_distance
                assert diff % 2 == 0
                for _ in range(diff // 2):
                    yield from way[1]
                yield from way[2]
            else:
                # connection is None, so this is the last visited battery
                if not battery.battery_field.free_fields_around:
                    # battery.charge must be 1
                    # there is no free field around this field, any neighbor field of battery.battery_field.pos will do
                    yield next(battery.battery_field.pos.neighbor_fields(0, self.size[0], 0, self.size[1]))
                elif len(battery.battery_field.free_fields_around) == 1:
                    # battery.charge must be <= 2 because there's only one free field around this battery field
                    yield battery.battery_field.free_fields_around[0]
                    if battery.charge == 2:
                        # any neighbor field of battery.battery_field.free_fields_around[0] is OK
                        yield next(battery.battery_field.free_fields_around[0].neighbor_fields(0, self.size[0],
                                                                                               0, self.size[1]))
                else:
                    if type(battery.battery_field.free_fields_around[0]) == TeleportationField:
                        # free_fields_around contains a pair of teleportation fields (which means that battery_field has
                        # a teleportation field as a neighbor)
                        assert battery.battery_field.free_fields_around[1] is None
                        tp_field = battery.battery_field.free_fields_around[0]
                        gen = itertools.cycle([["tp", tp_field.pos], ["tp", tp_field.paired_field.pos]])
                        yield next(gen)[1]  # yield first free field without "tp"
                        for _ in range(battery.charge-1):
                            yield from next(gen)
                    elif type(battery.battery_field.free_fields_around[1]) == TeleportationField:
                        # only second free field is teleportation field
                        yield battery.battery_field.free_fields_around[0]
                        if battery.charge > 1:
                            tp_field = battery.battery_field.free_fields_around[1]
                            gen = itertools.cycle([["tp", tp_field.pos], ["tp", tp_field.paired_field.pos]])
                            yield next(gen)[1]
                            for _ in range(battery.charge-2):
                                yield from next(gen)
                    else:
                        # both free fields are no teleportation field
                        gen = itertools.cycle(battery.battery_field.free_fields_around)
                        # yield neighbor fields until battery is empty
                        for _ in range(battery.charge):
                            yield next(gen)


def output_way(way, robot):
    if way:
        print("start at field", robot.robot_battery_field.pos, "with charge",
              robot.robot_battery_field.initial_battery.charge)
        if len(way) > 1:
            connection = way[0][1]
            print("go to field", connection.new_battery.battery_field.pos, "and")
            print("  drop battery with charge", connection.new_battery.charge, "on this field")
            for battery, connection in way[1:]:
                print("  grab battery with charge", battery.charge)
                if connection is not None:  # last connection is None
                    print("go to field", connection.new_battery.battery_field.pos, "and")
                    print("  drop battery with charge", connection.new_battery.charge, "on this field")
    else:
        print("did not find way")


DIRECTORY = "beispieldaten"
DIRECTORY_SVG = "beispieldaten_svg"
DIRECTORY_SOLUTIONS = "solutions"


def process_file(filename, do_print=False):
    print("\n############")
    print(filename)
    path = os.path.join(DIRECTORY, filename)
    robot = Robot.from_file(path)
    graph = robot.save_battery_field_graph(name=filename + "_fields", directory=DIRECTORY_SVG, format="svg")
    graph.render()
    robot.create_fake_batteries()
    input("Press enter")
    t1 = time.perf_counter_ns()
    way = robot.find_way(do_print=do_print)
    t2 = time.perf_counter_ns()
    print(f"time: {(t2 - t1) / 1e+9:.4f}s")
    output_way(way, robot)
    graph = robot.save_graph(way=way, name=filename, directory=DIRECTORY_SVG, format="svg")
    graph.render()

    # save a field with the found way
    with open(path, "r") as f:
        drawing = save_svg(f, solution=list(robot.generate_full_way(way)) if way else None)
        drawing.filename = os.path.join(DIRECTORY_SOLUTIONS, filename + ".svg")
        drawing.save(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(epilog="Saves graphviz-graphs to 'beispieldaten_svg' and solutions to 'solutions'")
    parser.add_argument("filename", default=None, type=str, metavar="filename", nargs="?",
                        help="Filename in directory 'beispieldaten' to process. If not specified, processes "
                             "stromrallye0.txt ... stromrallye5.txt")
    parser.add_argument("--print", dest="do_print", action="store_true", help="show output while doing backtracking")
    args = parser.parse_args()
    if args.filename is not None:
        process_file(args.filename, args.do_print)
    else:
        # beispieldaten
        for i in range(0, 6):
            process_file(f"stromrallye{i}.txt", args.do_print)
            input("Press enter")
