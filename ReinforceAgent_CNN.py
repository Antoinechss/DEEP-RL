import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from collections import deque
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# ----------------------------
# Tune hyperparameters
# ----------------------------
gamma = 0.99
lr = 0.0001
num_episodes = 500

# ----------------------------
# CNN Policy Network
# ----------------------------
class CNNPolicyNetwork(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(CNNPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

# ----------------------------
# REINFORCE Agent
# ----------------------------
class ReinforceAgent:
    def __init__(self, env, gamma, lr):
        self.env = env
        self.gamma = gamma
        self.lr = lr

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy = CNNPolicyNetwork(input_channels=4, action_dim=self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def preprocess_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs["image"]
        img = Image.fromarray(obs).convert("L").resize((84, 84))
        return np.array(img)

    def generate_episode(self, render=False):
        log_probs = []
        rewards = []

        raw_obs = self.env.reset()[0]
        state = self.preprocess_obs(raw_obs)
        frame_stack = deque([state] * 4, maxlen=4)

        done = False
        truncated = False

        while not (done or truncated):
            if render:
                self.env.render()
                import time
                time.sleep(0.05)

            stacked_state = np.stack(frame_stack, axis=0)
            stacked_state = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0)

            action_probs = self.policy(stacked_state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            log_probs.append(action_dist.log_prob(action))

            obs, reward, done, truncated, _ = self.env.step(action.item())
            next_state = self.preprocess_obs(obs)
            rewards.append(reward)
            frame_stack.append(next_state)

        return log_probs, rewards

    def compute_discounted_returns(self, rewards):
        G = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            G.insert(0, discounted_sum)
        return torch.tensor(G, dtype=torch.float32)

    def update_policy(self, log_probs, returns):
        loss = sum(-log_prob * G for log_prob, G in zip(log_probs, returns))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, save_path="reinforce_agent.pth", log_dir="runs/reinforce"):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"Old TensorBoard logs removed from {log_dir}")

        writer = SummaryWriter(log_dir=log_dir)
        best_reward = -float("inf")

        for episode in range(num_episodes):
            log_probs, rewards = self.generate_episode()
            returns = self.compute_discounted_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            self.update_policy(log_probs, returns)

            total_reward = sum(rewards)
            writer.add_scalar("Total Reward", total_reward, episode)

            if episode % 10 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), save_path)
                print(f"Model saved at Episode {episode} with Reward {best_reward}")

        writer.close()

    def load_model(self, save_path="reinforce_agent.pth"):
        self.policy.load_state_dict(torch.load(save_path))
        print("Model loaded successfully!")

# ----------------------------
# Training environment Setup
# ----------------------------
def make_train_env():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    return env

# ----------------------------
# Manual Test
# ----------------------------

# Training phase
train_env = make_train_env()
agent = ReinforceAgent(train_env, gamma=gamma, lr=lr)
agent.train(num_episodes=500)
train_env.close()

# Testing phase (visualize with GUI)
gui_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
agent.env = gui_env
agent.generate_episode(render=True)
gui_env.close()