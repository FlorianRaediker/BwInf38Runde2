"""
Aufgabe 1: Stromrallye

Dieses Skript ist Teil der Einsendung für die

    2. Runde des
    38. Bundeswettbewerbs Informatik

von

    Florian Rädiker.

Teilnahme-ID: 52570
"""
import itertools
import random
from typing import NamedTuple, Tuple, TextIO, Sequence

import svgwrite.container
import svgwrite.shapes
import svgwrite.text
import svgwrite.path


class Vector2(NamedTuple):
    x: int
    y: int

    def __str__(self):
        return f"({self.x}, {self.y})"

    def compact_str(self):
        return str(self.x) + "," + str(self.y)

    def __iter__(self):
        """
        Yields first y, then x. Instead of array[pos.y, pos.x], you can now write array[pos]
        """
        yield self.y
        yield self.x

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def up(self) -> "Vector2":
        return Vector2(self.x, self.y + 1)

    def down(self) -> "Vector2":
        return Vector2(self.x, self.y - 1)

    def right(self) -> "Vector2":
        return Vector2(self.x + 1, self.y)

    def left(self) -> "Vector2":
        return Vector2(self.x - 1, self.y)

    def neighbor_fields(self, /, min_x=None, max_x=None, min_y=None, max_y=None):
        if min_y is None or self.y != min_y:
            yield self.down()
        if max_x is None or self.x != max_x:
            yield self.right()
        if min_x is None or self.x != min_x:
            yield self.left()
        if max_y is None or self.y != max_y:
            yield self.up()


def read_uncommented_lines(file: TextIO):
    while line := file.readline():
        s = line.lstrip()
        if s:
            if not s.startswith("#"):
                yield s.split("#", 1)[0].rstrip()
            elif s.startswith("#!"):
                yield s


class StromrallyeParser:
    def __init__(self, file: TextIO):
        self._lines = read_uncommented_lines(file)
        self._size = None
        self._index_origin = 1
        self._parsed_size = False
        self._parsed_robot_battery = False
        self._parsed_batteries = False
        self._parsed_labels = False
        self._parsed_solution = False

    def _next(self):
        return next(self._lines)

    def size(self) -> Tuple[int, int]:
        if self._parsed_size:
            raise ValueError("size has already been parsed")
        self._parsed_size = True
        size_str = self._next()
        if "=" in size_str:
            size_str, self._index_origin = size_str.split("=", 1)
            self._index_origin = int(self._index_origin)
        if "," in size_str:
            # size consists of two sizes (width and height)
            width, height = size_str.split(",", 1)
            self._size = (int(width), int(height))
        else:
            size = int(size_str)
            self._size = (size, size)
        return self._size

    def _read_teleportation_field(self, line: str):
        assert line.startswith("T")
        line = line[1:]
        if ":" in line:
            line, label = line.split(":", 1)
        else:
            label = None
        # line represents teleportation fields
        if (c := line.count(",")) != 3:
            raise ValueError(f"Expected ',' three times in teleportation field line, found {c} occurrences: "
                             f"{repr(line)}")
        x1, y1, x2, y2 = line.split(",")
        pos1 = self._parse_position(x1 + "," + y1)
        pos2 = self._parse_position(x2 + "," + y2)
        return pos1, pos2, label

    def _read_battery(self, line: str):
        assert not line.startswith("T")
        if ":" in line:
            line, label = line.split(":", 1)
        else:
            label = None
        if (c := line.count(",")) != 2:
            raise ValueError(f"Expected ',' two times in battery line, found {c} occurrences: {repr(line)}")
        pos, charge = line.rsplit(",", 1)
        pos = self._parse_position(pos)
        try:
            charge = int(charge)
        except ValueError:
            raise ValueError(f"Expected int, got {repr(charge)}") from None
        assert 0 <= pos.x < self._size[0] and 0 <= pos.y < self._size[1], \
            f"battery coordinates are not within limits: {repr(line)}"
        return pos, charge, label

    def _read_battery_or_tp_field_line(self, line: str):
        if line.startswith("T"):
            return "tp_field", self._read_teleportation_field(line)
        return "battery", self._read_battery(line)

    def _parse_position(self, text: str):
        if (c := text.count(",")) != 1:
            raise ValueError(f"Expected ',' 1 time in position, found {c} occurrences: {repr(text)}")
        x, y = (coord for coord in text.split(","))
        try:
            x = int(x)
        except ValueError:
            raise ValueError(f"Expected int, got {repr(x)}") from None
        try:
            y = int(y)
        except ValueError:
            raise ValueError(f"Expected int, got {repr(y)}") from None
        return Vector2(x - self._index_origin, y - self._index_origin)

    def robot_battery(self):
        if not self._parsed_size:
            raise ValueError("Need to parse size (.size()) first")
        if self._parsed_robot_battery:
            raise ValueError("robot battery has already been parsed")
        self._parsed_robot_battery = True
        return self._read_battery(self._next())

    def batteries(self):
        if not self._parsed_robot_battery:
            raise ValueError("Need to parse robot battery (.robot_battery()) first")
        if self._parsed_batteries:
            raise ValueError("batteries have already been parsed")
        battery_count_str = self._next()
        if battery_count_str == "*":
            # parse the following lines with batteries, battery count is unknown
            for line in self._lines:
                if line == "-":  # '-' marks end of battery list (or EOF)
                    break
                yield self._read_battery_or_tp_field_line(line)
        else:
            try:
                for _ in range(1, int(battery_count_str) + 1):
                    yield self._read_battery_or_tp_field_line(self._next())
            except StopIteration:
                raise ValueError("Expected more batteries") from None
        self._parsed_batteries = True

    def labels(self):
        if not self._parsed_batteries:
            raise ValueError("Need to parse batteries (.batteries()) first")
        if self._parsed_labels:
            raise ValueError("labels have already been parsed")
        try:
            line = self._next()
        except StopIteration:
            self._parsed_labels = True
            return
        if line != "LABELS":
            raise ValueError(f"Expected '-' or EOF, got {repr(line)}")
        for line in self._lines:
            if line == "-":  # '-' marks end of labels list (or EOF)
                break
            if ":" not in line:
                raise ValueError(f"Expected ':' in label line, got {repr(line)}")
            pos, label = line.split(":", 1)
            pos = self._parse_position(pos)
            yield pos, label
        self._parsed_labels = True

    def solution(self):
        if not self._parsed_labels:
            raise ValueError("Need to parse labels (.labels()) first")
        if self._parsed_solution:
            raise ValueError("solution has already been parsed")
        self._parsed_solution = True
        SOLUTION = "#!solution="
        try:
            line = self._next()
        except StopIteration:
            return None
        if not line.startswith(SOLUTION):
            raise ValueError(f"Expected '#!solution' or EOF, got {repr(line)}")
        return (self._parse_position(pos) for pos in line[len(SOLUTION):].split(";"))

    def end(self):
        if not self._parsed_batteries:
            # batteries MUST be parsed, otherwise reading the file would make no sense at all ...
            raise ValueError("Need to parse batteries (.batteries()) first")
        # ... but not parsing labels or solution is OK
        if not self._parsed_labels:
            for _ in self.labels():
                pass
        if not self._parsed_solution:
            self.solution()
        try:
            line = self._next()
        except StopIteration:
            return
        else:
            raise ValueError(f"Expected EOF, got {repr(line)}")


def save_svg(file: TextIO, cell_size=100, font_size=40, solution: Sequence[Vector2] = None, draw_labels=True,
             draw_solution_from_file=False, use_positions_instead_of_labels=True) -> svgwrite.Drawing:
    grid_stroke_width = cell_size * 0.03

    parser = StromrallyeParser(file)
    width, height = parser.size()

    group = svgwrite.container.Group()
    drawing = svgwrite.Drawing(size=(width * cell_size + grid_stroke_width, height * cell_size + grid_stroke_width))
    group.translate(grid_stroke_width / 2, grid_stroke_width / 2)
    drawing.add(group)

    # grid
    grid_color = "#aaaaaa"
    grid = svgwrite.container.Group()
    group.add(grid)
    grid.add(svgwrite.shapes.Rect((0, 0), (width * cell_size, height * cell_size), stroke=grid_color,
                                  stroke_width=grid_stroke_width, fill="none"))

    # vertical lines
    for x in range(1, width):
        grid.add(svgwrite.shapes.Line((x * cell_size, 0), (x * cell_size, height * cell_size),
                                      stroke=grid_color, stroke_width=grid_stroke_width))
    # horizontal lines
    for y in range(1, height):
        grid.add(svgwrite.shapes.Line((0, y * cell_size), (width * cell_size, y * cell_size),
                                      stroke=grid_color, stroke_width=grid_stroke_width))

    robot_pos, robot_charge, robot_label = parser.robot_battery()

    half_cell_size = cell_size / 2

    group.add(svgwrite.shapes.Circle(
        (robot_pos.x * cell_size + half_cell_size, robot_pos.y * cell_size + half_cell_size),
        r=half_cell_size - grid_stroke_width*2, fill="green"))

    def add_label(pos, text, fill):
        kwargs = dict(x=[pos.x * cell_size + half_cell_size - cell_size / 5],
                      y=[pos.y * cell_size + half_cell_size - cell_size / 3],
                      font_size=str(font_size // 2),
                      text_anchor="middle",
                      alignment_baseline="central")
        if fill:
            kwargs["fill"] = fill
        group.add(svgwrite.text.Text(text, **kwargs))

    for type_, info in itertools.chain({("battery", (robot_pos, robot_charge, robot_label))}, parser.batteries()):
        if type_ == "battery":
            pos, charge, label = info
            group.add(svgwrite.text.Text(str(charge),
                                         x=[pos.x * cell_size + half_cell_size],
                                         y=[pos.y * cell_size + half_cell_size],
                                         font_size=str(font_size),
                                         text_anchor="middle",
                                         alignment_baseline="central"))
        else:
            assert type_ == "tp_field"
            pos, pos2, label = info
            for p in (pos, pos2):
                radius = half_cell_size - grid_stroke_width*4
                while radius > 0.5:
                    group.add(svgwrite.shapes.Circle(
                        (p.x * cell_size + half_cell_size, p.y * cell_size + half_cell_size),
                        r=radius, stroke="black", fill="none", stroke_width=grid_stroke_width/1.5)
                    )
                    radius -= grid_stroke_width*2 - random.random()*3 + 1.5
            group.add(svgwrite.shapes.Line(
                (pos.x * cell_size + half_cell_size, pos.y * cell_size + half_cell_size),
                (pos2.x * cell_size + half_cell_size, pos2.y * cell_size + half_cell_size),
                stroke_width="2", stroke="black")
            )
            if use_positions_instead_of_labels:
                add_label(pos2, str(pos2.x) + "," + str(pos2.y), "#777777")
        if label is None or use_positions_instead_of_labels:
            label = str(pos.x) + "," + str(pos.y)
            fill = "#777777"
        else:
            fill = None
        add_label(pos, label, fill)

    if draw_labels:
        for pos, label in parser.labels():
            add_label(pos, label, None)
    else:
        for _ in parser.labels():
            pass

    def draw_way(way, color="orange", stroke_dasharray="", stroke_width="3", first_position=None):
        kwargs = dict(stroke=color, stroke_width=stroke_width, fill="none")
        if stroke_dasharray:
            kwargs["stroke_dasharray"] = stroke_dasharray

        positions = [(pos.x * cell_size + half_cell_size + random.random() * half_cell_size - half_cell_size / 2,
                     pos.y * cell_size + half_cell_size + random.random() * half_cell_size - half_cell_size / 2)
                     for pos in way]
        if first_position:
            positions[0] = first_position
        group.add(svgwrite.shapes.Polyline(
            positions, **kwargs
            )
        )
        return positions[-1]

    if solution:
        i = -1
        pos = solution[0]
        next_position = (pos.x * cell_size + half_cell_size + random.random() * half_cell_size - half_cell_size / 2,
                         pos.y * cell_size + half_cell_size + random.random() * half_cell_size - half_cell_size / 2)
        while True:
            old_i = i
            try:
                i = solution.index("tp", i+1)
            except ValueError:
                draw_way(solution[i+1:], first_position=next_position)
                break
            assert solution[i+1] != "tp", "['tp', 'tp'] is not allowed in way"
            next_position = draw_way(solution[old_i+1:i], first_position=next_position)
            next_position = draw_way([solution[i-1], solution[i+1]], "purple", "5,5", "2", next_position)
    if draw_solution_from_file:
        if s := parser.solution():
            draw_way(s, "seagreen")

    parser.end()

    return drawing
