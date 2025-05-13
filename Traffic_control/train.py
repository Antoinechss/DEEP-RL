from sumo_env import SumoIntersectionEnv
from reinforce_agent import ReinforceAgent

# Import the intersection sumo.cfg file and convert it into a gym environment
env = SumoIntersectionEnv("/Users/antoinechosson/Desktop/intersection/1tls_2x2.sumocfg",use_gui=False)

# Link and train reinforce agent in env
agent = ReinforceAgent(env, gamma=0.99, lr=0.001)
agent.policy.apply(agent._init_weights)
agent.train(num_episodes=2000)
