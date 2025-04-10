# INTEGRATION OF REINFORCE INTO TRAFIC SIMULATION

Commands 

Observe initial intersection without control (replace with path to realistic_intersection folder): 

	sumo-gui -c /Users/antoinechosson/Desktop/realistic_intersection/intersection.sumocfg


Locate into project location :

  	cd ~/PycharmProjects/SUMO_RL

Train and save your reinforce agent : 

	python train.py

Visualize cumulated rewards over episodes with tensorboard tools: 

	tensorboard --logdir runs/reinforce

Observe the trained agent in [SUMO GUI](https://sumo.dlr.de/docs/sumo-gui.html) : 

	python observe.py 



