# pylint: disable=too-many-locals,too-many-arguments
import os

import fire
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map  # pylint: disable=only-importing-modules-is-allowed

from cvrp_experiments import belief_state, data, types, visualization, cvrp

OUTDIR = "cvrp_solutions"


def main(
    logs: str,
    timestep: int,
) -> None:
  os.makedirs(OUTDIR, exist_ok=True)
  logs = data.read_logs(logs)

  if timestep == -1:
    idx_and_logs = list(enumerate(logs))
    process_map(plot_and_save, idx_and_logs, max_workers=8)
  else:
    plot_and_save((timestep, logs[timestep]))


def plot_and_save(idx_and_log: tuple[int, str]) -> None:
  plt.clf()
  idx, log = idx_and_log
  raw_data = data.parse_log_line(log)
  extracted_data = extract_data(raw_data)
  vrp_solver = cvrp.VrpSolver(raw_data, True)
  vrp_solution = vrp_solver.solve_with_path()
  outpath = os.path.join(OUTDIR, f"vrp_solution_{idx}.png")
  generate_figure(*extracted_data, vrp_solution)
  plt.savefig(outpath)


def extract_data(
    raw_data: dict
) -> tuple[
    types.Robot,
    list[types.Robot],
    types.Path,
    list[types.Path],
    list[types.Cell],
    list[float],
    list[types.Connection],
]:
  current_robot = types.Robot.from_dict(raw_data["robots"][0])
  other_robots = [types.Robot.from_dict(i) for i in raw_data["robots"][1:]]
  global_path = types.Path.from_dict(raw_data["global_path"])
  other_robot_global_paths = [types.Path.from_dict(i) for i in raw_data["other_robot_global_paths"]]
  cells = [types.Cell.from_dict(i) for i in raw_data["cells"]]
  times_since_last_update = raw_data["time_since_last_update"]
  connections = [types.Connection.from_dict(c) for c in raw_data["connections"] if c != '...']
  return current_robot, other_robots, global_path, other_robot_global_paths, cells, times_since_last_update, connections


def generate_figure(
    current_robot: types.Robot,
    other_robots: list[types.Robot],
    global_path: types.Path,
    other_robot_global_paths: list[types.Path],
    cells: list[types.Cell],
    times_since_last_update: list[float],
    connections: list[types.Connection],
    cvrp_solution: types.Path,
) -> None:
  belief_states = []
  for robot, path, time in zip(other_robots, other_robot_global_paths, times_since_last_update):
    belief_states.append(belief_state.BeliefState(robot, path, time / 1000 * 2))
  aggregated_belief_state = belief_state.AggregatedBeliefState(belief_states)

  visualization.plot_heatmap(aggregated_belief_state, [-20, 80, -20, 80])

  for path in other_robot_global_paths:
    visualization.plot_path(path, "r")
  visualization.plot_path(global_path, "b")
  visualization.plot_path(cvrp_solution, "#002200", "#BBFFBB")

  visualization.plot_robot(current_robot)
  for robot, time in zip(other_robots, times_since_last_update):
    visualization.plot_robot(robot, time / 1000)

  for cell in cells:
    visualization.plot_cell(cell)
  ax = plt.gca()
  ax.set_xlim([-20, 80])
  ax.set_ylim([-20, 80])


if __name__ == "__main__":
  fire.Fire(main)
