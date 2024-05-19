# pylint: disable=no-member,only-importing-modules-is-allowed,too-few-public-methods,too-many-instance-attributes
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from cvrp_experiments import belief_state, types


class VrpSolver:

  def __init__(self, data: dict) -> None:
    self._current_robot = types.Robot.from_dict(data["robots"][0])
    self._other_robots = [types.Robot.from_dict(i) for i in data["robots"][1:]]
    self._other_robot_global_paths = [types.Path.from_dict(i) for i in data["other_robot_global_paths"]]
    self._times_since_last_update = data["time_since_last_update"]
    self._connections = types.Connections.from_dict(data)
    self._cell_ids = self._get_connected_cell_ids(data)
    self._cells = self._get_connected_cells(data)
    self._aggregated_belief_state = self._calc_aggregated_belief_state()
    self._num_vehicles = 1
    self._depot_indices = [1]
    self._end_indicies = [0]
    self._distance_matrix_size = len(self._cell_ids) + self._num_vehicles + 1

  def solve(self) -> list[int]:
    distance_matrix = self._calc_distance_matrix()
    likelihoods_at_nodes = self._calc_likeliehood_at_nodes()
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

    def likelihood_callback(from_index):
      from_node = manager.IndexToNode(from_index)
      return likelihoods_at_nodes[from_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    likelihood_callback_index = routing.RegisterUnaryTransitCallback(likelihood_callback)
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        1000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        "distance",
    )
    routing.AddDimension(
      likelihood_callback_index,
      0,  # null capacity slack
      1,
      True,  # start cumul to zero
      "likelihood",
    )
    routing.SetArcCostEvaluatorOfAllVehicles(likelihood_callback_index)
    distance_dimension = routing.GetDimensionOrDie("distance")
    distance_dimension.SetGlobalSpanCostCoefficient(0)
    likelihood_dimension = routing.GetDimensionOrDie("likelihood")
    likelihood_dimension.SetGlobalSpanCostCoefficient(100)
    # Add disjunction, allows nodes to be skipped
    for node in range(2, self._distance_matrix_size):
      routing.AddDisjunction([manager.NodeToIndex(node)], 10)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
      return self._get_solution(manager, routing, solution)
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

  def _get_solution(self, manager, routing, solution) -> list[int]:
    vrp_solution = []
    vehicle_id = 0
    index = routing.Start(vehicle_id)
    plan_output = f"Route for vehicle {vehicle_id}:\n"
    route_distance = 0
    route_likelihood = 0
    while not routing.IsEnd(index):
      node_index = manager.IndexToNode(index)
      plan_output += f"{node_index} -> "
      if node_index == 1:
        vrp_solution.append(self._current_robot.robot_id)
      elif node_index > 1:
        vrp_solution.append(self._cell_ids[node_index - self._num_vehicles - 1])
      previous_index = index
      index = solution.Value(routing.NextVar(index))
      distance_dimension = routing.GetDimensionOrDie("distance")
      route_distance += distance_dimension.GetTransitValue(previous_index, index, vehicle_id)
      likelihood_dimension = routing.GetDimensionOrDie("likelihood")
      route_likelihood += likelihood_dimension.GetTransitValue(previous_index, index, vehicle_id)
    node_index = manager.IndexToNode(index)
    plan_output += f"{node_index}\n"
    plan_output += f"Distance of the route: {route_distance}m\n"
    plan_output += f"Likelihood of vehicle intersecting belief state {vehicle_id}: {route_likelihood}\n"
    print(plan_output)
    print(vrp_solution)
    return vrp_solution

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

  def _calc_likeliehood_at_nodes(self) -> list[int]:
    likelihoods_at_nodes = [0 for _ in range(self._num_vehicles + 1)]
    for cell in self._cells:
      likelihood = self._aggregated_belief_state.get_likelihood(cell.position)
      likelihoods_at_nodes.append(int(likelihood * 100))
    return likelihoods_at_nodes

  def _get_connected_cell_ids(self, data: dict) -> list[int]:
    node_ids = data["cell_or_robot_ids"]
    is_node_robot = data["is_node_robot"]
    cell_ids = [node_id for node_id, is_node_robot in zip(node_ids, is_node_robot) if not is_node_robot]
    connected_cell_ids = [i for i in cell_ids if self._connections.is_node_connected(i, False)]
    return connected_cell_ids

  def _get_connected_cells(self, data:dict) -> list[types.Cell]:
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
