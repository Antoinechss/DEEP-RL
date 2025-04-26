import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import csv

# ----------------------------
# Tuning agent's hyperparameters
# ----------------------------

gamma = 0.99 # reward decay over time
lr = 0.001 # learning rate
num_episodes = 2000 # max number of episodes

# ----------------------------
# MLP Policy Network
# ----------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,128)  # creating fully connected layers
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state)) # computing rewards
        action_probs = F.softmax(self.fc2(x), dim=-1) # translating rewards into a probability distribution
        return action_probs

# ----------------------------
# REINFORCE Agent class
# ----------------------------

class ReinforceAgent:
    def __init__(self, env, gamma, lr):
        self.env = env
        self.gamma = gamma
        self.lr = lr

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.phase_counts = Counter()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def generate_episode(self, render=False):
        log_probs = []
        rewards = []

        state = self.env.reset()[0]
        done = False
        truncated = False

        while not (done or truncated):
            if render:
                self.env.render()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = self.policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            self.phase_counts[action.item()] += 1
            log_probs.append(action_dist.log_prob(action))
            state, reward, done, truncated, _ = self.env.step(action.item())
            rewards.append(reward)


        return log_probs, rewards

    def compute_discounted_returns(self, rewards):
        G = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            G.insert(0, discounted_sum)
        return torch.tensor(G, dtype=torch.float32)

    def update_policy(self, log_probs, returns):
        """
        Updates agent's policy after every episode using advantage and normalization.
        """
        # Baseline: moyenne des returns
        baseline = torch.mean(returns)

        # Avantage = return - baseline
        advantages = returns - baseline

        # Normalisation des avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Loss = somme des log_probs * avantages
        loss = -torch.sum(torch.stack([
            log_prob * advantage
            for log_prob, advantage in zip(log_probs, advantages)
        ]))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, save_path="reinforce_agent.pth", log_dir="runs/reinforce"):
        # Réinitialise les logs TensorBoard


        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"Old TensorBoard logs removed from {log_dir}")

        # Réinitialise le fichier KPI CSV
        with open("kpis.csv", mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Episode", "WaitingTime", "Delay", "QueueLength", "Volume"])
            print("kpis.csv has been reset.")

        writer = SummaryWriter(log_dir=log_dir)
        best_reward = -float("inf") # initialize best reward as default

        for episode in range(num_episodes):
            log_probs, rewards = self.generate_episode()
            returns = self.compute_discounted_returns(rewards)
            self.update_policy(log_probs, returns)

            total_reward = sum(rewards)
            writer.add_scalar("Total Reward", total_reward, episode)

            # KPI LOGGING
            kpis = self.env.get_kpis()
            writer.add_scalar("KPI/WaitingTime", kpis["waiting_time"], episode)
            writer.add_scalar("KPI/Delay", kpis["delay"], episode)
            writer.add_scalar("KPI/QueueLength", kpis["queue_length"], episode)
            writer.add_scalar("KPI/Volume", kpis["volume"], episode)

            with open("kpis.csv", mode="a", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([
                    episode,
                    kpis["waiting_time"],
                    kpis["delay"],
                    kpis["queue_length"],
                    kpis["volume"]
                ])

            if episode % 10 == 0:
                print(f"Episode {episode} | Reward: {total_reward:.2f}")
                print(f"Phase usage so far: {dict(self.phase_counts)}")

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), save_path)
                print(f"Model saved at Episode {episode} with Reward {best_reward}")

        writer.close() # exit metrics logging window


    def load_model(self, save_path="reinforce_agent.pth"):
        """
        Saves the trained agent as a reinforce_agent file
        """
        self.policy.load_state_dict(torch.load(save_path))
        print("Model loaded successfully!")
