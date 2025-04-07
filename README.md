# DEEP REINFORCEMENT LEARNING
Implementation of a reinforce agent for traffic control at road intersections 

This project aims at implementing a REINFORCE agent and visualize its effectiveness in various applications. 

**STEP #1** :Implement a generic REINFORCE and test it over gym environments 

- [`ReinforceAgent_MLP.py`](models/ReinforceAgent_MLP.py) : Simple 2 layer MLP version for discrete array input situations. Tested on [`CartPole-v1`](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

- [`ReinforceAgent_CNN.py`](models/ReinforceAgent_CNN.py) : CNN version for discrete tensor input (image treatment). Tested on [`MiniGrid-Empty-5x5-v0`](https://minigrid.farama.org/environments/minigrid/empty/)

You can monitor reward with tensorboard by running in terminal : 
$ tensorboard --logdir=runs/reinforce

**STEP #2** : Integration into SUMO environment for traffic control at an intersection 
