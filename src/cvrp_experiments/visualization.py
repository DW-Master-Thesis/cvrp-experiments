# pylint: disable=too-many-locals
import matplotlib.pyplot as plt
import numpy as np

from cvrp_experiments import types, belief_state


def plot_robot(robot: types.Robot, time_since_last_update: float | None = None) -> None:
  color = 'b' if time_since_last_update is None else 'r'
  ax = plt.gca()
  ax.scatter(
      [robot.position.x],
      [robot.position.y],
      edgecolors=color,
      marker="^",
      s=115,
      linewidths=2,
      c="#0000",
  )
  if time_since_last_update is None:
    return
  ax.scatter(
      [robot.state_estimation.x],
      [robot.state_estimation.y],
      c=color,
      marker="2",
      s=60,
      linewidths=2,
  )
  ax.plot(
      [robot.position.x, robot.state_estimation.x],
      [robot.position.y, robot.state_estimation.y],
      c=color,
      linestyle='--',
      linewidth=1,
  )
  circle = plt.Circle(
      (robot.position.x, robot.position.y),
      time_since_last_update * 2,
      color="k",
      linestyle='--',
      fill=False,
  )
  ax.add_patch(circle)


def plot_cell(cell: types.Cell, plot_connection_point: bool = False) -> None:
  ax = plt.gca()
  ax.scatter([cell.position.x], [cell.position.y], color='k', marker='x')
  if plot_connection_point:
    ax.scatter([cell.connection_point.x], [cell.connection_point.y], color='r', marker='x')


def plot_path(path: types.Path, color: str = "#AAAAAA") -> None:
  ax = plt.gca()
  x = [position.x for position in path.positions]
  y = [position.y for position in path.positions]
  ax.plot(x, y, marker=",", color=color)


def plot_heatmap(
    belief_state_: belief_state.BeliefState | belief_state.BeliefState,
    limits: list[float],
  ) -> None:
  ax = plt.gca()
  xmin, xmax, ymin, ymax = limits
  y_arr, x_arr = np.meshgrid(
    np.linspace(ymin, ymax, 100),
    np.linspace(xmin, xmax, 100),
  )
  z_arr = np.zeros_like(x_arr)
  for i in range(x_arr.shape[0]):
    for j in range(x_arr.shape[1]):
      z_arr[i, j] = belief_state_.get_likelihood(types.Position(x_arr[i, j], y_arr[i, j], 0))
  zmin, zmax = np.min(z_arr), np.max(z_arr)
  # ax.pcolormesh(x_arr, y_arr, z_arr, cmap='Blues', vmin=zmin, vmax=zmax)
  ax.pcolormesh(x_arr, y_arr, z_arr, shading='auto', cmap='Reds', vmin=zmin, vmax=zmax)
