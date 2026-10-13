import numpy as np
from gymnasium.utils.env_checker import check_env

from src import MyRobotEnv, TARGET_LOW, TARGET_HIGH

env = MyRobotEnv()
print(f"obs space {env.observation_space.shape}, action space {env.action_space.shape}")

# 1. Gymnasium API contract: spaces, dtypes, reset/step signatures, seed determinism
check_env(env, skip_render_check=True)
print("1. gymnasium check_env passed")

# 2. Target is re-sampled every reset, always inside the box
targets = []
for _ in range(5):
    env.reset()
    targets.append(env.target.copy())
targets = np.array(targets)
print("2. targets over 5 resets:\n", targets.round(3))
assert len(np.unique(targets, axis=0)) == 5, "target did not change between resets"
assert np.all((targets >= TARGET_LOW) & (targets <= TARGET_HIGH)), "target outside box"

# 3. Same seed -> same target
env.reset(seed=123); t1 = env.target.copy()
env.reset(seed=123); t2 = env.target.copy()
assert np.array_equal(t1, t2), "seeding does not control the target"
print("3. reset(seed=123) twice gives the same target:", t1.round(3))

# 4. First obs uses the real end-effector position, not the zeroed one
obs, _ = env.reset()
ee_start = np.array([0.391, 0.0, 0.226])
assert np.allclose(obs[-3:], env.target - ee_start, atol=1e-3), "first obs has stale end-effector position"
print("4. first obs target-ee vector:", obs[-3:].round(3), "(ee at", ee_start, ")")

# 5. Reward is consistent with the observation's distance vector
obs, reward, *_ = env.step(env.action_space.sample())
assert np.isclose(reward, -np.linalg.norm(obs[-3:]), atol=1e-5)
print(f"5. reward {reward:.4f} == -|obs[-3:]| {-np.linalg.norm(obs[-3:]):.4f}")
