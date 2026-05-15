""""
CartPole Training & Evaluation (PyTorch + Gymnasium)
"""
from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

# 导入所有 Agent
from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_a2c import A2CAgent, A2CConfig  # <--- 新增
from agents.cartpole_ppo import PPOAgent, PPOConfig  # <--- 新增
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

# ----------------------------------------------------
# 📌 在这里修改你想训练的算法！
# Options: "dqn", "a2c", "ppo"
# ----------------------------------------------------
CURRENT_ALGORITHM = "ppo" 
# ----------------------------------------------------

def get_model_path(algo_name):
    return os.path.join(MODEL_DIR, f"cartpole_{algo_name}.torch")

def train(num_episodes: int = 500, terminal_penalty: bool = True) -> object:
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 根据算法名称选择 Agent
    print(f"[Train] Initializing Agent: {CURRENT_ALGORITHM.upper()}")
    if CURRENT_ALGORITHM == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif CURRENT_ALGORITHM == "a2c":
        agent = A2CAgent(obs_dim, act_dim, cfg=A2CConfig())
    elif CURRENT_ALGORITHM == "ppo":
        # PPO 训练通常需要更多回合，或者去掉 terminal_penalty 会更稳
        agent = PPOAgent(obs_dim, act_dim, cfg=PPOConfig())
    else:
        raise ValueError(f"Unknown algorithm: {CURRENT_ALGORITHM}")

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            action = agent.act(state) # act 内部处理 exploration

            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # DQN 需要这个惩罚，但 Policy Gradient (PPO/A2C) 有时不需要
            # 这里保留以保持一致性，但你可以尝试在 PPO 中去掉它
            if terminal_penalty and done and steps < 500:
                reward = -1.0
            
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 学习步骤
            agent.step(state, action, reward, next_state, done)

            state = next_state

            if done:
                # 获取探索率用于打印 (DQN特有，其他算法设为 None 或 0)
                eps = getattr(agent, "exploration_rate", 0.0)
                print(f"Run: {run}, Algo: {CURRENT_ALGORITHM}, Eps: {eps:.3f}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()
    save_path = get_model_path(CURRENT_ALGORITHM)
    agent.save(save_path)
    print(f"[Train] Model saved to {save_path}")
    return agent


def evaluate(algorithm: str = "dqn", episodes: int = 5, render: bool = True, fps: int = 60):
    model_path = get_model_path(algorithm)
    
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}. Train it first!")
        return

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 实例化对应的 Agent
    if algorithm == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm == "a2c":
        agent = A2CAgent(obs_dim, act_dim, cfg=A2CConfig())
    elif algorithm == "ppo":
        agent = PPOAgent(obs_dim, act_dim, cfg=PPOConfig())
    
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
            if dt > 0: time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    print(f"[Eval] Average: {np.mean(scores):.2f}")


if __name__ == "__main__":
    # 自动根据算法调整参数
    if CURRENT_ALGORITHM == "a2c":
        print("检测到 A2C 算法：增加训练轮数，关闭死亡惩罚...")
        # A2C 需要更多练习 (1500轮)，且不喜欢死亡惩罚 (False)
        agent = train(num_episodes=1500, terminal_penalty=False)
    else:
        # DQN 和 PPO 500轮通常就够了
        agent = train(num_episodes=1500, terminal_penalty=True)
    
    evaluate(algorithm=CURRENT_ALGORITHM, episodes=100, render=True)