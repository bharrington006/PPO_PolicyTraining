from pathlib import Path

from stable_baselines3 import PPO

from src import MyRobotEnv

HERE = Path(__file__).resolve().parent

env = MyRobotEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(HERE / "logs"), seed=0)
model.learn(total_timesteps=500_000)
model.save(HERE / "ppo_reach")
