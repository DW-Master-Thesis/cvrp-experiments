import json
import os

import matplotlib.pyplot as plt
import numpy as np

MARKERS = ['s', '^', 'D', 'v', '*', 'X', 'P']

DATA_DIR = 'tsp_solution_data/'

FILENAME_TO_METHOD = {
    "BASELINE": "baseline",
    "max-reward_min-dist_best-initial-strategy": "Max. reward,\nMin. dist,\nBest initial strategy",
    "max-reward_min-dist_default-initial-strategy": "Max. reward,\nMin. dist,\nDefault initial strategy",
    "max-reward_no-dist_best-initial-strategy": "Max. reward,\nBest initial strategy",
    "max-reward_no-dist_default-initial-strategy": "Max. reward,\nDefault initial strategy",
    "min-cost_no-dist_default-initial-strategy": "Min. cost,\nDefault initial strategy",
}


def _get_solution_data_filepaths() -> list[str]:
  return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]


def _load_solution_data(filepath: str) -> dict:
  with open(filepath, 'r', encoding="utf-8") as f:
    return json.load(f)


def _calculate_reward_ratio(solution_data: dict) -> list[float]:
  res = []
  for reward, penalty in zip(solution_data['rewards'], solution_data['penalties']):
    # if penalty == 0:
    #   continue
    if reward + penalty == 0:
      res.append(0)
    else:
      res.append(reward / (reward + penalty))
  return res


def _calc_sum_first_n_reward_ratio(solution_data: dict, n: int) -> list[int]:
  res = []
  for reward_evolution, reward, penalty in zip(
      solution_data['rewards_evolution'], solution_data['rewards'], solution_data['penalties']
  ):
    total_reward = reward + penalty
    if not reward_evolution:
      res.append(0)
      continue
    # if _calc_avg(reward_evolution) > 950:
    #   continue
    if len(reward_evolution) == 1:
      res.append(reward_evolution[0] / total_reward)
      continue
    n = min(n, len(reward_evolution))
    res.append(sum(reward_evolution[:n]) / total_reward)
  return res


def _calc_mean_and_std(data: list[int] | list[float]) -> tuple[float, float]:
  if not data:
    return 0, 0
  return np.mean(data), np.std(data)


def plot_reward_ratio(solution_data: dict) -> None:
  plt.clf()
  filename_to_label = {
      "BASELINE": "baseline (VRP)",
      "min-cost_no-dist_default-initial-strategy": "Min. cost",
      "max-reward_no-dist_default-initial-strategy": "Max. reward",
      "max-reward_min-dist_default-initial-strategy": "Max. reward,\nMin. dist",
  }
  methods = list(filename_to_label.keys())
  reward_ratio_avgs = [solution_data[method]['reward_ratio_avg'] for method in methods]
  reward_ratio_stds = [solution_data[method]['reward_ratio_std'] for method in methods]
  # sort by reward_ratio_avg
  methods = [x for _, x in sorted(zip(reward_ratio_avgs, methods))]
  methods = [filename_to_label[method] for method in methods]
  reward_ratio_avgs.sort()
  fig, ax = plt.subplots()
  ax.barh(methods, reward_ratio_avgs, label=methods, xerr=reward_ratio_stds)
  ax.set_xlabel('Average percentage of reward collected [%]')
  ax.set_title('Average percentage of reward collected vs TSP formulation')
  # Make labels visible
  plt.tight_layout()
  plt.savefig('reward_ratio_avg.png')


def plot_reward_first_5(solution_data: dict) -> None:
  plt.clf()
  filename_to_y_label = {
      "BASELINE": "baseline",
      "max-reward_no-dist_default-initial-strategy": "Max. reward",
      "max-reward_no-dist_best-initial-strategy": "Max. reward",
      "max-reward_min-dist_default-initial-strategy": "Max. reward,\nMin. dist",
      "max-reward_min-dist_best-initial-strategy": "Max. reward,\nMin. dist",
  }
  default_methods = [
      "BASELINE",
      "max-reward_no-dist_default-initial-strategy",
      "max-reward_min-dist_default-initial-strategy",
  ]
  best_methods = [
      "max-reward_no-dist_best-initial-strategy",
      "max-reward_min-dist_best-initial-strategy",
  ]
  y = np.arange(len(default_methods))
  width = 0.35
  x_data_default = [solution_data[method]['reward_first_5_avg'] for method in default_methods]
  x_data_best = [0] + [solution_data[method]['reward_first_5_avg'] for method in best_methods]
  y_label = [filename_to_y_label[method] for method in default_methods]
  # sort by reward_ratio_avg
  fig, ax = plt.subplots()
  ax.barh(y - width / 2, x_data_default, width, label='Default initial strategy')
  ax.barh(y + width / 2, x_data_best, width, label='Best initial strategy')
  ax.set_yticks(y)
  ax.set_yticklabels(y_label)
  ax.set_xlabel('Average percentage of reward collected in first 5 steps [%]')
  ax.set_title('Average percentage of reward collected in first 5 steps\nvs First solution strategy')
  ax.legend()
  # Make labels visible
  plt.tight_layout()
  plt.savefig('reward_first_5.png')


def plot_distance_vs_reward(solution_data: dict) -> None:
  method_to_label = {
      "BASELINE": "baseline",
      "max-reward_no-dist_best-initial-strategy": "Max. reward",
      "max-reward_min-dist_best-initial-strategy": "Max. reward,\nMin. dist",
  }
  methods = list(method_to_label.keys())
  plt.clf()
  ax = plt.gca()
  for method in methods:
    data = solution_data[method]
    distances = data['distances']
    rewards = data['rewards']
    label = method_to_label[method]
    marker = MARKERS.pop()
    ax.scatter(distances, rewards, label=label, marker=marker)
  ax.set_xlabel('Distance [m]')
  ax.set_ylabel('Reward')
  ax.set_xlim(0, 1000)
  ax.set_title('Distance vs Reward')
  ax.legend()
  plt.tight_layout()
  plt.savefig('distance_vs_reward.png')


def main():
  solution_data_filepaths = _get_solution_data_filepaths()
  solution_data = {}
  for filepath in solution_data_filepaths:
    filename = os.path.basename(filepath)
    solution_data[filename] = _load_solution_data(filepath)
    solution_data[filename]['reward_ratio'] = _calculate_reward_ratio(solution_data[filename])
    reward_ratio_avg, reward_ratio_std = _calc_mean_and_std(solution_data[filename]['reward_ratio'])
    solution_data[filename]['reward_ratio_avg'] = reward_ratio_avg * 100
    solution_data[filename]['reward_ratio_std'] = reward_ratio_std * 100
    solution_data[filename]['reward_first_5'] = _calc_sum_first_n_reward_ratio(solution_data[filename], 5)
    reward_first_5_avg, reward_first_5_std = _calc_mean_and_std(solution_data[filename]['reward_first_5'])
    solution_data[filename]['reward_first_5_avg'] = reward_first_5_avg * 100
    solution_data[filename]['reward_first_5_std'] = reward_first_5_std * 100
    print(f'{filename}: {solution_data[filename]["reward_ratio_avg"]}')
    print(f'{filename}: {solution_data[filename]["reward_first_5_avg"]}')
  plot_reward_ratio(solution_data)
  plot_reward_first_5(solution_data)
  plot_distance_vs_reward(solution_data)


if __name__ == '__main__':
  main()
