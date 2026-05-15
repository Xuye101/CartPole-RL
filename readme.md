算法代码位置：.\agents
模型位置：.\models


要运行cartpole_a2c.py cartpole_dqn.py cartpole_ppo.py 只需在train.py中，将CURRENT_ALGORITHM 改为相应算法名 


例如：
# Options: "dqn", "a2c", "ppo"
# ----------------------------------------------------
CURRENT_ALGORITHM = "a2c" 

python train.py


# ----------------------------------------------------
offline RL运行命令
确保已完成以下准备工作：
有训练好的PPO 专家模型（用于生成演示数据），默认路径为 models/cartpole_ppo.torch（由 train.py 训练生成，训练时需将 CURRENT_ALGORITHM 设为 "ppo"）。
已生成专家演示数据（expert_data.npz），由 generate_data.py 生成（依赖上面的 PPO 模型）。

生成专家演示数据（若未生成）python generate_data.py

运行离线训练脚本（训练 BC 模型）python train_offline.py