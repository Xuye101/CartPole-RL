import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BCConfig:
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100  # 离线训练多少轮
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

class BCAgent:
    def __init__(self, obs_dim, act_dim, cfg: BCConfig = None):
        self.cfg = cfg or BCConfig()
        self.device = torch.device(self.cfg.device)
        self.model = BCNet(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss() # 分类问题的标准损失函数

    def train_offline(self, states, actions):
        """核心：离线训练函数"""
        # 1. 准备数据
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        
        dataset = TensorDataset(states_t, actions_t)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        print("Start Offline Training (Behavior Cloning)...")
        self.model.train()
        
        for epoch in range(self.cfg.epochs):
            total_loss = 0
            for batch_s, batch_a in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_s)
                loss = self.criterion(logits, batch_a)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.cfg.epochs}, Loss: {total_loss/len(loader):.4f}")

    def act(self, state, evaluation_mode=True):
        """在线测试用"""
        self.model.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            logits = self.model(state_t)
            action = torch.argmax(logits, dim=1).item()
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))