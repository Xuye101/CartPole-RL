import numpy as np
import gymnasium as gym
import torch
from agents.cartpole_ppo import PPOAgent, PPOConfig  # 导入你的满分 PPO

def generate_expert_data(model_path, output_file="expert_data.npz", episodes=50):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 加载你的满分 PPO 模型
    agent = PPOAgent(obs_dim, act_dim, cfg=PPOConfig())
    agent.load(model_path)
    print(f"Loaded expert model from {model_path}")

    all_states = []
    all_actions = []
    total_reward = 0

    for ep in range(episodes):
        state, _ = env.reset(seed=1000+ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0
        
        while not done:
            # 使用 PPO 预测动作
            action = agent.act(state, evaluation_mode=True)
            
            # 存下来！(State, Action)
            all_states.append(state.squeeze()) # 存成 1D 数组
            all_actions.append(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
        
        total_reward += steps
        print(f"Episode {ep+1}: Steps {steps}")

    # 保存成文件
    np.savez(output_file, states=np.array(all_states), actions=np.array(all_actions))
    print(f"Data saved to {output_file}. Average Score: {total_reward/episodes}")
    print(f"Total samples: {len(all_states)}")

if __name__ == "__main__":
    # 确保路径对，用你之前跑出来的满分 PPO 模型
    generate_expert_data("models/cartpole_ppo.torch", "expert_data.npz", episodes=200)