import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass

# -----------------------------
# A2C Hyperparameters (Robust Version)
# -----------------------------
GAMMA = 0.5
# 学习率恢复到 1e-3，因为我们去掉了激进的归一化，需要大学习率来驱动
LR = 1e-3            
ENTROPY_BETA = 0.01 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class A2CConfig:
    gamma: float = GAMMA
    lr: float = LR
    entropy_beta: float = ENTROPY_BETA
    device: str = DEVICE

class A2CAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: A2CConfig = None):
        self.cfg = cfg or A2CConfig()
        self.device = torch.device(self.cfg.device)
        
        # 【核心修复1】拆分 Actor 和 Critic
        # 以前是共享层，容易打架。现在分开，计算图互不干扰，极其稳定。
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # 【核心修复2】权重初始化
        # 保证一开始左右动作概率是 50/50，防止开局就“坍塌”到 9 分
        self._init_weights(self.actor)
        self._init_weights(self.critic)

        # 两个网络的参数一起优化
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=self.cfg.lr
        )
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def _init_weights(self, module):
        for m in module:
            if isinstance(m, nn.Linear):
                # 正交初始化，RL 的标配，比 Xavier 更稳
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        
        probs = self.actor(state_t)
        value = self.critic(state_t) # 存下来算 Loss
        
        dist = Categorical(probs)

        if evaluation_mode:
            action = torch.argmax(probs, dim=1).item()
        else:
            action = dist.sample()
            
            self.log_probs.append(dist.log_prob(action))
            self.values.append(value)
            self.entropies.append(dist.entropy())
            
            action = action.item()
        return action

    def step(self, state, action, reward, next_state, done):
        if done:
            self.rewards.append(reward)
            self.update()
        else:
            self.rewards.append(reward)

    def update(self):
        R = 0
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.cfg.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 【核心修复3】温和的缩放
        # 之前 (x - mean) / std 会导致除以 0 爆炸。
        # 现在简单除以 100，让数值保持在小范围即可。
        returns = returns / 100.0 
        
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        entropies = torch.stack(self.entropies)
        
        # 维度修正
        if values.dim() == 0: values = values.unsqueeze(0)

        # Advantage (Detached)
        advantage = returns - values.detach()

        # Loss Calculation
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + self.cfg.entropy_beta * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        
        # 依然保留裁剪，双重保险
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        
        self.optimizer.step()

        # 每次 update 完必须清空，否则就会报 "backward second time" 错误
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def save(self, path):
        # 保存两个网络的权重
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])