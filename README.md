# CVRP experiments

This repository experiments with the constrained vehicle routing problem (VRP).
It investigates whether it is suitable to solve the global planning problem in TARE, when there are multiple robots, and there exists a time delay between communication.

## Problem setup
There are N robots exploring an unknown environment.
Each robot communicates with a central server, sending updates about their respecitve environment.
The server aggregates the received information and broadcasts it to other agents after some time delay.
This delay simulates the time taken to perform loop closure in multi-robot SLAM.

Between each update, a robot keeps track of a belief state of the position of other robots.
The belief state is calculated based on the position and global plan of the other robots at the previous update.

This belief state is used in calculating the cost of visiting some frontiers during exploration.
(The cells are represented as cells in TARE.)
For example, there is a high likelihood that an agent is positioned close to a frontier, the cost to explore that frontier is scaled higher, to discourage redundant exploration.

Finally, an agent plans its global path using the constrained VRP.
It is given a limited exploration budget.
Thus it prioritises frontiers that are close-by and likely to be further from other agents.
