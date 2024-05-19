import dataclasses

import numpy as np


@dataclasses.dataclass
class Position:
  x: float
  y: float
  z: float

  @staticmethod
  def from_dict(data: dict) -> "Position":
    return Position(data["x"], data["y"], data["z"])

  def distance_to(self, other_position: "Position") -> float:
    return np.sqrt((self.x - other_position.x) ** 2 + (self.y - other_position.y) ** 2)


@dataclasses.dataclass
class Robot:
  position: Position
  state_estimation: Position
  robot_id: int

  @staticmethod
  def from_dict(data: dict) -> "Robot":
    return Robot(
      Position.from_dict(data["position"]),
      Position.from_dict(data["state_estimation"]),
      data["id"]
    )


@dataclasses.dataclass
class Cell:
  position: Position
  connection_point: Position
  cell_id: int

  @staticmethod
  def from_dict(data: dict) -> "Cell":
    return Cell(
      Position.from_dict(data["position"]),
      Position.from_dict(data["connection_point"]),
      data["id"]
    )


@dataclasses.dataclass
class Path:
  positions: list[Position]

  @staticmethod
  def from_dict(data: dict) -> "Path":
    positions_ = []
    for pose in data["poses"]:
      positions_.append(Position.from_dict(pose["pose"]["position"]))
    return Path(positions_)

  @staticmethod
  def from_vrp_solution(vrp_solution: list[int], cells: list[Cell]) -> "Path":
    positions_ = []
    for node_id in vrp_solution:
      cell = next(c for c in cells if c.cell_id == node_id)
      positions_.append(cell.position)
    return Path(positions_)

  def distance_to(self, other_position: Position) -> float:
    if len(self.positions) == 0:
      return 0
    if len(self.positions) == 1:
      return self.positions[0].distance_to(other_position)

    def _is_p3_between_p1_and_p2(p1: Position, p2: Position, p3: Position) -> bool:
      return (p1.x < p3.x and p3.x < p2.x) or (p1.x > p3.x and p3.x > p2.x) \
        and (p1.y < p3.y and p3.y < p2.y) or (p1.y > p3.y and p3.y > p2.y)

    def _calc_p3_prime(p1: Position, p2: Position, p3: Position) -> Position:
      m = (p2.y - p1.y) / (p2.x - p1.x)
      b = p1.y - m * p1.x
      y3_ = m * p3.x + b
      return Position(p3.x, y3_, p3.z)

    p3 = other_position
    distances = []
    for i in range(len(self.positions) - 1):
      p1, p2 = self.positions[i], self.positions[i+1]
      if (p2.x - p1.x) != 0:
        p3_ = _calc_p3_prime(p1, p2, p3)
      else:
        p3_ = Position(p1.x, p3.y, p3.z)
      if _is_p3_between_p1_and_p2(p1, p2, p3_):
        distances.append(p3_.distance_to(p3))
      else:
        distances.append(min([p1.distance_to(p3), p2.distance_to(p3)]))
    return min(distances)

  def extend(self, path: "Path") -> "Path":
    self.positions.extend(path.positions)


@dataclasses.dataclass
class Connection:
  from_node_id: int
  is_from_node_robot: bool
  to_node_id: int
  is_to_node_robot: bool
  distance: int
  path: Path

  @staticmethod
  def from_dict(data: dict) -> "Connection":
    return Connection(
        data["from_node_id"],
        data["is_from_node_robot"],
        data["to_node_id"],
        data["is_to_node_robot"],
        data["distance"],
        Path.from_dict(data["path"])
    )

  def connects_nodes(
      self,
      node_id_1: int,
      is_node_id_1_robot: bool,
      node_id_2: int,
      is_node_id_2_robot: bool,
  ) -> bool:
    node_type_matches_forward = self._node_type_matches(is_node_id_1_robot, is_node_id_2_robot)
    node_type_matches_backward = self._node_type_matches(is_node_id_2_robot, is_node_id_1_robot)

    if node_type_matches_forward:
      return self.from_node_id == node_id_1 and self.to_node_id == node_id_2
    if node_type_matches_backward:
      return self.from_node_id == node_id_2 and self.to_node_id == node_id_1
    return False

  def _node_type_matches(self, node_type_1: bool, node_type_2: bool) -> bool:
    return node_type_1 == self.is_from_node_robot and node_type_2 == self.is_to_node_robot


@dataclasses.dataclass
class Connections:
  connections: list[Connection]

  @staticmethod
  def from_dict(data: dict) -> "Connections":
    return Connections([Connection.from_dict(connection) for connection in data["connections"]])

  def get_connection_distance(
      self,
      from_node_id: int,
      is_from_node_robot: bool,
      to_node_id: int,
      is_to_node_robot: bool,
  ) -> int:
    for connection in self.connections:
      if connection.connects_nodes(from_node_id, is_from_node_robot, to_node_id, is_to_node_robot):
        return connection.distance
    return 9999

  def is_node_connected(
      self,
      node_id: int,
      is_node_robot: bool,
  ) -> bool:
    for connection in self.connections:
      if connection.from_node_id == node_id and connection.is_from_node_robot == is_node_robot:
        return True
      if connection.to_node_id == node_id and connection.is_to_node_robot == is_node_robot:
        return True
    return False

  def get_path_between_nodes(
      self,
      from_node_id: int,
      is_from_node_robot: bool,
      to_node_id: int,
      is_to_node_robot: bool,
  ) -> Path:
    for connection in self.connections:
      if connection.connects_nodes(from_node_id, is_from_node_robot, to_node_id, is_to_node_robot):
        return connection.path
    return Path([])
