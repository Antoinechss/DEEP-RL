from sumo_env import SumoIntersectionEnv
from reinforce_agent import ReinforceAgent

# Load the environment with GUI for visualization
env = SumoIntersectionEnv("/Users/antoinechosson/Desktop/intersection/1tls_2x2.sumocfg", use_gui=True)

# Load trained agent
agent = ReinforceAgent(env, gamma=0.99, lr=1e-4)
agent.load_model("reinforce_agent.pth")

# Observe one episode
env.use_gui = True
agent.env = env
agent.generate_episode(render=True)

env.close()

