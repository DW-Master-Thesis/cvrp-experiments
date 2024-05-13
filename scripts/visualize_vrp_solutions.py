import fire
import matplotlib.pyplot as plt

from cvrp_experiments import types, data, visualization

def main(
    logs: str,
    timestep: int,
) -> None:
  logs = data.read_logs(logs)
  raw_data = data.parse_log_line(logs[timestep])
  current_robot = types.Robot.from_dict(raw_data["robots"][0])
  other_robots = [types.Robot.from_dict(i) for i in raw_data["robots"][1:]]
  cells = [types.Cell.from_dict(i) for i in raw_data["cells"]]
  times_since_last_update = raw_data["time_since_last_update"]
  connections = [types.Connection.from_dict(c) for c in raw_data["connections"] if c != '...']

  for connection in connections:
    visualization.plot_path(connection.path)

  for other_robot_global_path in raw_data["other_robot_global_paths"]:
    other_robot_global_path = types.Path.from_dict(other_robot_global_path)
    visualization.plot_path(other_robot_global_path, "r")
  vrp_solution = types.Path.from_dict(raw_data["global_path"])
  visualization.plot_path(vrp_solution, "b")

  visualization.plot_robot(current_robot)
  for robot, time in zip(other_robots, times_since_last_update):
    visualization.plot_robot(robot, time / 1000)

  for cell in cells:
    visualization.plot_cell(cell)

  plt.savefig(f"vrp_solution_{timestep}.png")


if __name__ == "__main__":
  fire.Fire(main)
