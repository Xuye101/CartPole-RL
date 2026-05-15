import numpy as np
import gymnasium as gym
import os  # 新增：用于处理路径
from agents.cartpole_bc import BCAgent, BCConfig
# 导入ScoreLogger（根据你的实际路径调整导入方式）
from scores.score_logger import ScoreLogger  # 若路径不同可使用sys.path添加

def train_bc():
    # 1. 加载数据
    data = np.load("expert_data.npz")
    states = data['states']
    actions = data['actions']
    print(f"Loaded data: {len(states)} samples")

    # 2. 初始化BC Agent和日志记录器
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = BCAgent(obs_dim, act_dim)
    
    # 初始化ScoreLogger（指定环境名称，用于日志文件命名）
    logger = ScoreLogger("CartPole-v1-BC")  # 区分BC的日志

    # 3. 离线训练（修改训练过程以记录损失）
    print("Start Offline Training (Behavior Cloning)...")
    agent.model.train()
    states_t = torch.FloatTensor(states).to(agent.device)
    actions_t = torch.LongTensor(actions).to(agent.device)
    dataset = TensorDataset(states_t, actions_t)
    loader = DataLoader(dataset, batch_size=agent.cfg.batch_size, shuffle=True)

    for epoch in range(agent.cfg.epochs):
        total_loss = 0
        for batch_s, batch_a in loader:
            agent.optimizer.zero_grad()
            logits = agent.model(batch_s)
            loss = agent.criterion(logits, batch_a)
            loss.backward()
            agent.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        # 用ScoreLogger记录损失（这里将损失作为"分数"传入，方便复用日志功能）
        logger.add_score(avg_loss, epoch + 1)  # 第二个参数是当前轮次
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{agent.cfg.epochs}, Loss: {avg_loss:.4f}")

    # 4. 保存模型
    os.makedirs("models", exist_ok=True)
    agent.save("models/cartpole_bc.torch")
    print("BC Model saved.")
    
    return agent

def evaluate_bc(agent):
    # 5. 评估模型
    env = gym.make("CartPole-v1", render_mode="human")
    print("\nEvaluating BC Agent...")
    for i in range(5): # 可以改成100，测试100轮
        state, _ = env.reset(seed=2000 + i)
        state = np.reshape(state, (1, 4))
        done = False
        steps = 0
        while not done:
            action = agent.act(state)
            state, _, terminated, truncated, _ = env.step(action)
            state = np.reshape(state, (1, 4))
            done = terminated or truncated
            steps += 1
        print(f"Episode {i+1}: Score {steps}")

if __name__ == "__main__":
    # 确保导入torch（原代码漏了，补充上）
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    agent = train_bc()
    evaluate_bc(agent)