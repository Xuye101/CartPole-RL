import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass

# -----------------------------
# PPO Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 2e-3   # PPO 通常可以用大一点的学习率
EPS_CLIP = 0.2  # PPO 核心参数：截断范围
K_EPOCHS = 4   # 每次更新循环训练几次
ENTROPY_BETA = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PPOConfig:
    gamma: float = GAMMA
    lr: float = LR
    eps_clip: float = EPS_CLIP
    k_epochs: int = K_EPOCHS
    entropy_beta: float = ENTROPY_BETA
    device: str = DEVICE

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(), # PPO 常用 Tanh
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def get_action(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig = None):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.policy = PPOActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.policy_old = PPOActorCritic(obs_dim, act_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Buffer
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
        self.exploration_rate = 0 # PPO 是策略梯度，不需要 epsilon-greedy

    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if evaluation_mode:
            with torch.no_grad():
                probs = self.policy.actor(state_t)
                action = torch.argmax(probs, dim=1).item()
            return action
        else:
            with torch.no_grad():
                action, logprob, _ = self.policy_old.get_action(state_t)
            
            # 暂存用于 Update 的数据
            self.states.append(state_t)
            self.actions.append(action)
            self.logprobs.append(logprob)
            
            return action

    def step(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.is_terminals.append(done)
        
        # PPO 也是回合制更新比较方便
        if done:
            self.update()

    def update(self):
        # 1. 计算 Monte Carlo Returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.cfg.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 转换 Buffer 为 Tensor
        old_states = torch.cat(self.states).detach().view(-1, 4) # Hardcoded 4 dim for cartpole or use obs_dim
        old_actions = torch.tensor(self.actions, dtype=torch.float32, device=self.device).detach()
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=self.device).detach()

        # 2. PPO Update loop (K epochs)
        for _ in range(self.cfg.k_epochs):
            # 评估旧动作在新策略下的概率和价值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # 计算 Ratios
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算 Advantage
            advantages = rewards - state_values.detach()

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * nn.functional.mse_loss(state_values, rewards) - self.cfg.entropy_beta * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 3. 同步旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 4. 清空 Buffer
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())