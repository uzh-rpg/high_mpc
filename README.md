## High-MPC: Learning High-Level Policies for Model Predictive Control
This project contains the code for solving the problem of passing through a pendulum-like gate.
We make use of a high-level policy learning strategy to train a neural network high-level policy which 
can adaptively select decision variables (traversal time) for MPC. 
Based on the decision variable, the (perfect) penulum dynamics, and the quadrotor's state,
our high-mpc predicts a sequence control commands and states for the quadrotor.
The first control command is applied to the system, after which the optimization problem
is solved again in the next state.
Eventually, our algorithm manage to control the quadrotor to pass through the center of
the swinging gate, where we randomly initialized the state of the system. 


![High_MPC_Demo](docs/high_mpc_0.gif)


### Installation 

Clone the repo

```
git clone git@github.com:uzh-rpg/high_mpc.git
```

Installation Dependencies:

```
cd high_mpc
pip install -r requirements.txt
```

Add the repo path to your PYTHONPATH by adding the following to your ~/.bashrc

```
export PYTHONPATH=${PYTHONPATH}:/path/to/high_mpc
```

### Run Examples

