# 🤸‍♂️ CartPole-RL: Robust PPO & A2C Implementations in PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenAI Gym](https://img.shields.io/badge/OpenAI_Gym-000000?style=for-the-badge&logo=openai&logoColor=white)

This repository contains clean, robust, and optimized PyTorch implementations of two classic deep reinforcement learning algorithms: **PPO (Proximal Policy Optimization)** and **A2C (Advantage Actor-Critic)**, applied to the classic OpenAI Gymnasium `CartPole-v1` environment.

## 🌟 Key Highlights & Optimizations

Rather than blindly tuning hyperparameters, this project focuses on **critical thinking and architectural debugging** to address the notoriously unstable nature of policy gradient methods.

### 1. The A2C Stability Fixes (`cartpole_a2c.py`)
A naive implementation of A2C often suffers from early policy collapse (getting stuck at low scores like 9). To solve this, we implemented three core modifications:
*   **Actor-Critic Decoupling:** Separated the shared hidden layers between the Actor and Critic networks. This prevents gradient interference where the value loss overrides the policy loss.
*   **Orthogonal Initialization:** Applied orthogonal weight initialization to ensure the initial action probabilities are near 50/50, preventing premature convergence to a sub-optimal deterministic policy.
*   **Gentle Reward Scaling:** Replaced standard normalization `(x - mean) / std` with a gentle scaling factor (`returns / 100.0`). This eliminates the vanishing/exploding gradient problem caused by division by zero when the batch variance is extremely small in early episodes.

### 2. The PPO Sample Efficiency (`cartpole_ppo.py`)
To overcome A2C's poor sample efficiency, we implemented PPO with a clipped surrogate objective.
*   **Trust Region Updates:** By utilizing `eps_clip = 0.2`, the agent safely re-uses the same batch of transitions for `K_epochs = 4` updates without destroying the current policy.
*   **Entropy Regularization:** Included an entropy bonus (`beta = 0.01`) to encourage early exploration and prevent premature exploitation.

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
```bash
pip install torch numpy gymnasium
