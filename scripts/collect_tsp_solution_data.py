# pylint: disable=too-many-locals,too-many-arguments
import json
import os
from concurrent import futures

import fire
import tqdm

from cvrp_experiments import cvrp, data

OUTDIR = "tsp_solution_data"


def main(logs: str, output_filename: str) -> None:
  os.makedirs(OUTDIR, exist_ok=True)
  logs = data.read_logs(logs)

  distances: list[int] = []
  rewards: list[int] = []
  penalties: list[int] = []
  rewards_evolution: list[list[int]] = []

  pb = tqdm.tqdm(total=len(logs))
  with futures.ProcessPoolExecutor(max_workers=8) as executor:
    futures_ = [executor.submit(_solve_vrp, log) for log in logs]
    for future in futures.as_completed(futures_):
      distance, reward, penalty, reward_evolution = future.result()
      distances.append(distance)
      rewards.append(reward)
      penalties.append(penalty)
      rewards_evolution.append(reward_evolution)
      pb.update(1)

  print("Distances:", distances)
  print("Rewards:", rewards, f"({sum(rewards)})")

  with open(os.path.join(OUTDIR, output_filename), "w", encoding="utf-8") as f:
    json.dump({
        "distances": distances,
        "rewards": rewards,
        "penalties": penalties,
        "rewards_evolution": rewards_evolution,
    }, f, indent=2)


def _solve_vrp(log: str) -> tuple[int, int, int, list[int]]:
  raw_data = data.parse_log_line(log)
  vrp_solver = cvrp.VrpSolver(raw_data, True)
  _ = vrp_solver.solve_with_path()
  return vrp_solver.distance, vrp_solver.reward, vrp_solver.penalty, vrp_solver.reward_evolution


if __name__ == "__main__":
  fire.Fire(main)
