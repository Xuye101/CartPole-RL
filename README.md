# 🤸‍♂️ CartPole-RL: Robust Online & Offline Implementations

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenAI Gym](https://img.shields.io/badge/OpenAI_Gym-000000?style=for-the-badge&logo=openai&logoColor=white)

[**[中文版 README 在下方 / Chinese Version Below]**](#中文版简介)

This repository contains clean, robust, and optimized PyTorch implementations of classic deep reinforcement learning algorithms: **PPO**, **A2C**, **DQN**, and **Offline RL (Behavioral Cloning)**, applied to the classic OpenAI Gymnasium `CartPole-v1` environment.

## 📂 Project Structure
```text
.
├── agents/                 # Implementation of RL algorithms
│   ├── cartpole_a2c.py
│   ├── cartpole_dqn.py
│   └── cartpole_ppo.py
├── models/                 # Saved model weights
├── train.py                # Unified training entry point for Online RL
├── generate_data.py        # Script to generate expert data using trained PPO
├── train_offline.py        # Offline RL training (Behavioral Cloning)
└── README.md
```bash
pip install torch numpy gymnasium
