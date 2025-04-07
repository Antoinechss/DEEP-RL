**INTEGRATING REINFORCE AGENT TO A SUMO ENVIRONMENT**
The goal of the second part of this project is to use the REINFORCE algorithm to control traffic signals at an intersection. 
To do so I use the already built [DQN framework for SUMO](https://github.com/romainducrocq/DQN-ITSCwPD.git) by Ducrocq Romain, who implemented several DQN type algorithms. 

**Task** = transitionning from DQN to Policy Based : 

- Action selection : Sample from $\pi(a,s)$
- On policy training step (uses episode trajectory)
- Loss function = policy gradient
- Policy network (probabilities instead of values)
- Output = probability distribution over actions

**Integration approach**:

1. Adapt the Environment
The provided framework is tailored for the SUMO environment (traffic simulation). I first replace the MiniGrid gym environment setup in the generic impelmentation of reinforceCNN with a suitable SUMO environment from the DQN-ITSCwPD framework. In particular, ensuring the observation preprocessing (resizing, grayscale, stacking frames) in order to align with the visual inputs  from SUMO.

2. Replace DQN with REINFORCE
Replace the existing DQN agent (DQNAgent) with my ReinforceAgent. Adapting agent's initialization (state_dim, action_dim) according to the environment provided by SUMO.

3. Observation Handling
Verify the CNN input dimensions (input_channels) correspond to the visual data (frames stacked) provided by the SUMO environment.

4. Reward & Action Space
Update your reward and action mechanisms based on your traffic control objective (e.g., traffic waiting times, queue lengths, intersection throughput).

5. Policy Updates and Logging
Ensure TensorBoard logging and model-saving mechanisms from your REINFORCE script are correctly integrated within the training loop of the DQN-ITSCwPD project.
