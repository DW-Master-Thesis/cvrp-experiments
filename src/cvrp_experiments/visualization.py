# pylint: disable=too-many-locals
import matplotlib.pyplot as plt
import numpy as np

from cvrp_experiments import belief_state, types


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


def plot_path(path: types.Path, color: str = "#AAAAAA", end_color: str | None = None) -> None:
  ax = plt.gca()
  x = [position.x for position in path.positions]
  y = [position.y for position in path.positions]
  if end_color:
    colors = _calc_color_gradient(color, end_color, len(path.positions) - 1)
    for x1, y1, x2, y2, color in zip(x[:-1], y[:-1], x[1:], y[1:], colors):  # pylint: disable=redefined-argument-from-local
      ax.plot([x1, x2], [y1, y2], color=color)
  else:
    ax.plot(x, y, marker=",", color=color)


def _calc_color_gradient(start_color: str, end_color: str, num_colors: int) -> list[str]:
  r_start, g_start, b_start = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
  r_end, g_end, b_end = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
  r_step, g_step, b_step = (r_end - r_start) / num_colors, (g_end -
                                                            g_start) / num_colors, (b_end - b_start) / num_colors
  colors = []
  r, g, b = r_start, g_start, b_start
  for _ in range(num_colors):
    r, g, b = r + r_step, g + g_step, b + b_step
    colors.append(f"#{int(r):02X}{int(g):02X}{int(b):02X}")
  return colors


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
