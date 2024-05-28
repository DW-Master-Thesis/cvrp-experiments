# pylint: disable=no-member,only-importing-modules-is-allowed,too-few-public-methods,too-many-instance-attributes
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from cvrp_experiments import belief_state, types


class VrpSolver:

  def __init__(self, data: dict, silent_mode: bool = False, use_baseline_vrp_solution: bool = False) -> None:
    self._silent_mode = silent_mode
    self._use_baseline_vrp_solution = use_baseline_vrp_solution
    self._current_robot = types.Robot.from_dict(data["robots"][0])
    self._other_robots = [types.Robot.from_dict(i) for i in data["robots"][1:]]
    self._other_robot_global_paths = [types.Path.from_dict(i) for i in data["other_robot_global_paths"]]
    self._baseline_vrp_solution = []
    if use_baseline_vrp_solution and len(data["vrp_solution"]) > 0:
      self._baseline_vrp_solution = data["vrp_solution"][0]["route"]
    self._times_since_last_update = data["time_since_last_update"]
    self._connections = types.Connections.from_dict(data)
    self._cell_ids = self._get_connected_cell_ids(data)
    self._cells = self._get_connected_cells(data)
    self._aggregated_belief_state = self._calc_aggregated_belief_state()
    self._num_vehicles = 1
    self._depot_indices = [1]
    self._end_indicies = [0]
    self._distance_matrix_size = len(self._cell_ids) + self._num_vehicles + 1
    self.distance, self.reward, self.penalty, self.reward_evolution = 0, 0, 0, []

  def solve(self) -> list[int]:
    distance_matrix = self._calc_distance_matrix()
    node_rewards = self._calc_node_rewards()
    # node_costs = self._calc_node_costs()
    manager = pywrapcp.RoutingIndexManager(
        self._distance_matrix_size,
        self._num_vehicles,
        self._depot_indices,
        self._end_indicies,
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      return distance_matrix[from_node][to_node]

    def distance_and_reward_callback(from_index, to_index):
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      distance = distance_matrix[from_node][to_node]
      reward = node_rewards[to_node]
      if distance == 0:
        return 0
      return distance - reward // 10

    # def cost_callback(from_index):
    #   from_node = manager.IndexToNode(from_index)
    #   cost = int(100 * node_costs[from_node])
    #   return cost

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.AddDimension(
        distance_callback_index,
        0,  # no slack
        1000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        "distance",
    )
    transit_callback_index = routing.RegisterTransitCallback(distance_and_reward_callback)
    # routing.AddDimension(
    #     transit_callback_index,
    #     0,  # no slack
    #     10_000_000,  # vehicle maximum travel distance
    #     True,  # start cumul
    #     "distance_and_reward",
    # )
    # cost_callback_index = routing.RegisterUnaryTransitCallback(cost_callback)
    # routing.AddDimension(
    #   cost_callback_index,
    #   0,  # no slack
    #   1000,  # vehicle maximum travel distance
    #   True,  # start cumul to zero
    #   "cost",
    # )
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    # Add disjunction, allows nodes to be skipped
    for node in range(self._num_vehicles + 1, self._distance_matrix_size):
      routing.AddDisjunction([manager.NodeToIndex(node)], node_rewards[node])
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT

    if self._use_baseline_vrp_solution:
      return self._extract_baseline_solution(manager, routing)

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
      vrp_solution = self._extract_solution(manager, routing, solution)
      if not self._silent_mode:
        self._print_solution(manager, routing, vrp_solution)
      return self._vrp_ids_to_node_ids(vrp_solution)
    print("No solution found.")
    return []

  def solve_with_path(self) -> types.Path:
    vrp_solution = self.solve()
    path = types.Path([])
    for i in range(len(vrp_solution) - 1):
      from_node_id = vrp_solution[i]
      to_node_id = vrp_solution[i + 1]
      from_node_is_robot = i == 0
      path_between_nodes = self._connections.get_path_between_nodes(from_node_id, from_node_is_robot, to_node_id, False)
      path.extend(path_between_nodes)
    return path

  def _extract_solution(self, manager, routing, solution) -> list[int]:
    self.distance, self.reward, self.penalty, self.reward_evolution = 0, 0, 0, []
    node_rewards = self._calc_node_rewards()
    self.penalty = sum(node_rewards)
    vrp_solution = []
    index = routing.Start(0)
    distance_dimension = routing.GetDimensionOrDie("distance")
    while not routing.IsEnd(index):
      node_index = manager.IndexToNode(index)
      vrp_solution.append(node_index)
      previous_index = index
      index = solution.Value(routing.NextVar(index))
      distance_from_previous = distance_dimension.GetTransitValue(previous_index, index, 0)
      self.distance += distance_from_previous
      if node_index <= self._num_vehicles:
        continue
      reward = node_rewards[node_index]
      self.reward += reward
      self.reward_evolution.append(reward)
    self.penalty -= self.reward
    return vrp_solution

  def _vrp_ids_to_node_ids(self, vrp_indices: list[int]) -> list[int]:
    vrp_solution = []
    for node_idx in vrp_indices:
      if node_idx == 1:
        vrp_solution.append(self._current_robot.robot_id)
      elif node_idx > 1:
        vrp_solution.append(self._cell_ids[node_idx - self._num_vehicles - 1])
    return vrp_solution

  def _print_solution(self, manager, routing, vrp_indices: list[int]) -> None:  # pylint: disable=too-many-locals
    plan_output = ""
    distance_dimension = routing.GetDimensionOrDie("distance")
    for prev_node_idx, node_idx in zip(vrp_indices[:-1], vrp_indices[1:]):
      previous_index = manager.NodeToIndex(prev_node_idx)
      index = manager.NodeToIndex(node_idx)
      distance_from_previous = distance_dimension.GetTransitValue(previous_index, index, 0)
      plan_output += f"{prev_node_idx} ->({distance_from_previous}) "
    plan_output += f"{vrp_indices[-1]}\n"
    plan_output += f"Distance of the route: {self.distance}m\n"
    plan_output += f"Reward of the route: {self.reward}\n"
    plan_output += f"Penalty of the route: {self.penalty}\n"
    print(plan_output)
    print(self._vrp_ids_to_node_ids(vrp_indices))

  def _extract_baseline_solution(self, manager, routing) -> list[int]:
    node_rewards = self._calc_node_rewards()
    self.distance, self.reward, self.penalty, self.reward_evolution = 0, 0, 0, []
    self.penalty = sum(node_rewards)
    vrp_solution = []
    vrp_solution.append(1)
    if not self._baseline_vrp_solution:
      if not self._silent_mode:
        print("No baseline solution found.")
      return vrp_solution
    for i in self._baseline_vrp_solution[1:]:
      if i not in self._cell_ids:
        if not self._silent_mode:
          print(f"Node {i} not in connected cells")
      else:
        idx = self._cell_ids.index(i) + 2
        vrp_solution.append(idx)
    distance_dimension = routing.GetDimensionOrDie("distance")
    for prev_idx, idx in zip(vrp_solution[1:-1], vrp_solution[2:]):
      previous_index = manager.NodeToIndex(prev_idx)
      index = manager.NodeToIndex(idx)
      self.reward += node_rewards[idx]
      self.penalty -= node_rewards[idx]
      self.distance += distance_dimension.GetTransitValue(previous_index, index, 0)
      self.reward_evolution.append(node_rewards[idx])
    if not self._silent_mode:
      self._print_solution(manager, routing, vrp_solution)
    return self._vrp_ids_to_node_ids(vrp_solution)

  def _calc_distance_matrix(self) -> list[list[int]]:
    distance_matrix = [[0 for _ in range(self._distance_matrix_size)] for _ in range(self._distance_matrix_size)]
    for i, cell_id in enumerate(self._cell_ids):
      # Second column is from robot to cells
      distance_matrix[i + 2][1] = self._calc_connection_distance(
          self._current_robot.robot_id,
          True,
          cell_id,
          False,
      )
      # bottom right of matrix starting at second row/column is cells to cells
      for j in range(i):
        distance_matrix[i + 2][j + 2] = self._calc_connection_distance(
            cell_id,
            False,
            self._cell_ids[j],
            False,
        )
    # invert the matrix
    for i in range(self._distance_matrix_size):
      for j in range(i):
        distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

  def _calc_connection_distance(
      self,
      from_node_id: int,
      is_from_node_robot: bool,
      to_node_id: int,
      is_to_node_robot: bool,
  ) -> int:
    return self._connections.get_connection_distance(
        from_node_id,
        is_from_node_robot,
        to_node_id,
        is_to_node_robot,
    )

  def _calc_node_costs(self) -> list[float]:
    node_costs = [0 for _ in range(self._num_vehicles + 1)]
    for cell in self._cells:
      likelihood = self._aggregated_belief_state.get_likelihood(cell.position)
      likelihood = min(1, likelihood / 0.1)
      node_costs.append(likelihood)
    return node_costs

  def _calc_node_rewards(self) -> list[int]:
    node_costs = self._calc_node_costs()
    node_rewards = [int(1000 * (1 - likelihood)) for likelihood in node_costs]
    for i in range(self._num_vehicles + 1):
      node_rewards[i] = 0
    return node_rewards

  def _get_connected_cell_ids(self, data: dict) -> list[int]:
    node_ids = data["cell_or_robot_ids"]
    is_node_robot = data["is_node_robot"]
    cell_ids = [node_id for node_id, is_node_robot in zip(node_ids, is_node_robot) if not is_node_robot]
    connected_cell_ids = [i for i in cell_ids if self._connections.is_node_connected(i, False)]
    return connected_cell_ids

  def _get_connected_cells(self, data: dict) -> list[types.Cell]:
    all_cells = [types.Cell.from_dict(i) for i in data["cells"]]
    connected_cells = []
    for cell_id in self._cell_ids:
      connected_cells.append(next(c for c in all_cells if c.cell_id == cell_id))
    return connected_cells

  def _calc_aggregated_belief_state(self) -> belief_state.AggregatedBeliefState:
    belief_states = []
    for robot, path, time in zip(self._other_robots, self._other_robot_global_paths, self._times_since_last_update):
      belief_states.append(belief_state.BeliefState(robot, path, time / 1000 * 2))
    return belief_state.AggregatedBeliefState(belief_states)
