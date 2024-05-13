import numpy as np

from cvrp_experiments import types


class BeliefState:
  SIGMA = 5
  K1 = 1.0 / (SIGMA * np.sqrt(2 * np.pi))
  K2 = 1.0 / (2 * SIGMA**2)

  def __init__(self, robot: types.Robot, global_plan: types.Path, limit: float):
    self.robot = robot
    self.global_plan = global_plan
    self.limit = limit

  def update(self, robot: types.Robot, global_plan: types.Path, limit: float):
    self.robot = robot
    self.global_plan = global_plan
    self.limit = limit

  def get_likelihood(self, position: types.Position):
    dist_to_path = self.global_plan.distance_to(position)
    dist_to_robot = self.robot.position.distance_to(position)
    dist = min(dist_to_path, dist_to_robot)
    if dist > self.limit:
      return 0
    return self.K1 * np.exp(-self.K2 * dist**2)


class AggregatedBeliefState:  # pylint: disable=too-few-public-methods

  def __init__(self, belief_states: list[BeliefState]):
    self.belief_states = belief_states

  def get_likelihood(self, position: types.Position):
    likelihoods = [belief_state.get_likelihood(position) for belief_state in self.belief_states]
    if len(likelihoods) == 0:
      return 0
    return np.max(likelihoods)
