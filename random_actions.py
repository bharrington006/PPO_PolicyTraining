import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from src import MyRobotEnv

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--headless", action="store_true", help="skip the viewer, only print stats")
args = parser.parse_args()

env = MyRobotEnv()
model, data = env.model, env.data
env.action_space.seed(0)
obs, _ = env.reset(seed=0)

viewer = None if args.headless else mujoco.viewer.launch_passive(model, data)


def draw_target(viewer, pos):
    # Decoration only: user_scn geoms are rendered but not part of the physics
    viewer.user_scn.ngeom = 1
    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.015, 0, 0]), pos, np.eye(3).flatten(),
                        np.array([1, 0, 0, 0.5], dtype=np.float32))


qpos_min, qpos_max = data.qpos.copy(), data.qpos.copy()
max_qvel = 0.0
deepest_dist, deepest_pair = 0.0, None
episodes = 0
sim_time = 0.0
wall_start = time.perf_counter()

for _ in range(args.steps):
    time_before = data.time
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    sim_time += data.time - time_before

    qpos_min = np.minimum(qpos_min, data.qpos)
    qpos_max = np.maximum(qpos_max, data.qpos)
    max_qvel = max(max_qvel, np.abs(data.qvel).max())
    if data.ncon:
        i = np.argmin(data.contact.dist)
        if data.contact.dist[i] < deepest_dist:
            g1, g2 = data.contact.geom[i]
            deepest_dist = data.contact.dist[i]
            deepest_pair = f"{model.body(model.geom_bodyid[g1]).name} <-> {model.body(model.geom_bodyid[g2]).name}"

    if viewer:
        if not viewer.is_running():
            break
        draw_target(viewer, env.target)
        viewer.sync()
        # Pace to real time so one second on screen is one second of robot time
        time.sleep(max(0.0, sim_time - (time.perf_counter() - wall_start)))

    if terminated or truncated:
        episodes += 1
        obs, _ = env.reset()

print(f"Simulated {sim_time:.2f} s of robot time in {args.steps} steps; episodes finished: {episodes}")
print("Joint range explored vs. full range:")
for j in range(model.njnt):
    a = model.jnt_qposadr[j]
    lo, hi = model.jnt_range[j]
    print(f"  {model.joint(j).name:<14} [{qpos_min[a]:6.2f}, {qpos_max[a]:6.2f}] of [{lo:6.2f}, {hi:6.2f}]"
          f"  ({(qpos_max[a] - qpos_min[a]) / (hi - lo):4.0%})")
print(f"Max |qvel|: {max_qvel:.2f} rad/s")
print(f"Deepest contact: {deepest_dist * 1000:.1f} mm ({deepest_pair})" if deepest_pair else "No contacts")

if viewer:
    print("Viewer stays open so you can look around; close the window to exit.")
    while viewer.is_running():
        time.sleep(0.1)
    viewer.close()
