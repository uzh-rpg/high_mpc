## High-MPC: Learning High-Level Policies for Model Predictive Control

<!-- ![Method](docs/figures/MethodOverview.png) -->
<img src="docs/figures/MethodOverview.png" alt="drawing" style="width:500px;"/>


This project contains the code for solving the problem of passing through a pendulum-like gate.
We make use of a high-level policy learning strategy to train a neural network which 
can adaptively select decision variables (traversal time) for MPC. 
Based on the decision variable, the (perfect) penulum dynamics, and the quadrotor's state,
our high-mpc predicts a sequence control commands and states for the quadrotor.
The first control command is applied to the system, after which the optimization problem
is solved again in the next state.
Eventually, our algorithm manage to control the quadrotor to pass through the center of
the swinging gate, where we randomly initialized the state of the system.

Please find a list of demonstrations in [here](docs/gifs/README.md). 

![High_MPC_Demo](docs/gifs/high_mpc_trail2.gif)

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

### Run 

Standard MPC

```
cd high_mpc
python3 run_mpc.py
```

Learning a High-Level Policy

```
python3 run_highmpc.py 
```

Learning a Deep High-Level Policy

```
# collect training data for the MLP
python3 run_deep_highmpc.py --option 0

# train the deep high-level policy with pre-collected data
python3 run_deep_highmpc.py --option 1

# evaluate the performance with pre-trained deep high-level policy
python3 run_deep_highmpc.py --option 2
```

