# DEEP REINFORCEMENT LEARNING - Intelligent Traffic Signal Control with Partial Detection
Implementation of the REINFORCE Deep RL algorithm for traffic control at road intersections 

This project aims at implementing a REINFORCE agent and visualize its effectiveness in various applications. 

**STEP 1** :Implement a generic REINFORCE agent and test it over simple gym environments 

- [`ReinforceAgent_MLP.py`](models/ReinforceAgent_MLP.py) : Simple 2 layer MLP version for discrete array input situations. Tested on [`CartPole-v1`](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

- [`ReinforceAgent_CNN.py`](models/ReinforceAgent_CNN.py) : Convolutional Policy Network for image-based environments.
  - Uses a 3-layer CNN with frame stacking and softmax output for discrete actions.
  - Automatically preprocesses RGB observations into grayscale, resized frames.
  - Logs training progress to TensorBoard for easy visualization.
  - Tested on: [`MiniGrid-Empty-5x5-v0`](https://minigrid.farama.org/environments/minigrid/empty/)

To launch training and visualize performance (from terminal):
```bash
python ReinforceAgent_CNN.py
tensorboard --logdir=runs/reinforce
```


**STEP 2** : Create an intersection SUMO environment and integrate the reinforce agent for intelligent traffic control 

See [SUMO documentation](https://sumo.dlr.de/docs/index.html)

**Key Performance Indicators** (computed after each phase selection) : 

1/ **Accumulated waiting time** : total sum of cumulative waiting times since insertion for all incoming vehicles

2/ **Delay** : the total sum of immediate delays for all incoming vehicles

3/ **Queue length** : the total number of waiting incoming vehicles

4/ **Volume** : total number of incoming vehicles




