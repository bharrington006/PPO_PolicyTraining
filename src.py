from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco

MODEL_PATH = Path(__file__).resolve().parent.parent / "mujoco_playground/SO-ARM100/Simulation/SO101/scene.xml"

# Box the target is sampled from (meters, world frame). Every 2 cm cell in it was
# hit by the gripperframe site when sampling 500k random joint configurations.
TARGET_LOW = np.array([0.15, -0.15, 0.05])
TARGET_HIGH = np.array([0.30, 0.15, 0.25])

class MyRobotEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.target = np.zeros(3)

        # qpos + qvel + (target - end-effector) vector
        obs_dim = self.model.nq + self.model.nv + 3
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.target = self.np_random.uniform(TARGET_LOW, TARGET_HIGH)
        # mj_resetData zeroes site_xpos; recompute it so the first obs sees the real end-effector position
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _ee_pos(self):
        return self.data.site("gripperframe").xpos

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self.target - self._ee_pos(),
        ]).astype(np.float32)

    def _compute_reward(self):
        return -np.linalg.norm(self._ee_pos() - self.target)

    def _check_termination(self):
        return False  # your task-specific termination condition
