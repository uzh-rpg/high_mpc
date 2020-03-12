# High-MPC: Learning High-Level Policies for Model Predictive Control
This project contains the code for solving the problem of passing through a pendulum-like gate.
We make use of a high-level policy learning strategy to train a neural network high-level policy which 
can adaptively select decision variables (traversal time) for MPC. 
Based on the decision variable, the (perfect) penulum dynamics, and the quadrotor's state,
our high-mpc predicts a sequence control commands and states for the quadrotor.
The first control command is applied to the system, after which the optimization problem
is solved again in the next state.


### 

Dependencies:

* numpy 1.17.3
* casadi 3.5.1
* matplotlib 3.1.1
* tensorflow 2.1.0
