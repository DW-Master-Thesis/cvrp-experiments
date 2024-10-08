import os

import fire
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map  # pylint: disable=only-importing-modules-is-allowed

from cvrp_experiments import types, data, visualization

OUTDIR = "VRP"

def main(
    logs: str,
    timestep: int,
) -> None:
  os.makedirs(OUTDIR, exist_ok=True)
  logs = data.read_logs(logs)

  if timestep == -1:
    idx_and_logs = list(enumerate(logs))
    process_map(plot_and_save, idx_and_logs, max_workers=5)
  else:
    plot_and_save((timestep, logs[timestep]))


def plot_and_save(idx_and_log: tuple[int, str]) -> None:
  plt.clf()
  idx, log = idx_and_log
  raw_data = data.parse_log_line(log)
  generate_figure(raw_data)
  outpath = os.path.join(OUTDIR, f"vrp_solution_{idx}.png")
  plt.savefig(outpath, bbox_inches='tight', pad_inches=0.1)


def generate_figure(raw_data):
  current_robot = types.Robot.from_dict(raw_data["robots"][0])
  other_robots = [types.Robot.from_dict(i) for i in raw_data["robots"][1:]]
  cells = [types.Cell.from_dict(i) for i in raw_data["cells"]]
  times_since_last_update = raw_data["time_since_last_update"]

  # connections = [types.Connection.from_dict(c) for c in raw_data["connections"] if c != '...']
  # for connection in connections:
  #   visualization.plot_path(connection.path)

  for other_robot_global_path in raw_data["other_robot_global_paths"]:
    other_robot_global_path = types.Path.from_dict(other_robot_global_path)
    visualization.plot_path(other_robot_global_path, "r", label="Other robot's global path")
  vrp_solution = types.Path.from_dict(raw_data["global_path"])
  visualization.plot_path(vrp_solution, "b", label="Current robot's global path")

  visualization.plot_robot(current_robot)
  for robot, time in zip(other_robots, times_since_last_update):
    visualization.plot_robot(robot, time / 1000)

  for cell in cells:
    visualization.plot_cell(cell)
  if len(cells) > 0:
    visualization.plot_cell(cells[0], label="Cells to visit")

  plt.legend(loc='lower right')
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.gca().tick_params(axis='both', which='both', length=0)
  plt.title("Solution to the VRP problem")
  plt.tight_layout()
  ax = plt.gca()
  ax.set_xlim([-20, 80])
  ax.set_ylim([0, 100])


if __name__ == "__main__":
  fire.Fire(main)
