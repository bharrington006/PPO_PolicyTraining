import mujoco
import numpy as np

from src import MODEL_PATH

np.set_printoptions(precision=3, suppress=True)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("start  qpos =", data.qpos, " qvel =", data.qvel)
print("       gripperframe xyz =", data.site("gripperframe").xpos, "\n")

# Ask the shoulder_pan servo to go to 0.5 rad; the other five servos keep holding 0
data.ctrl[0] = 0.5

print(" time   qpos[0] (rad)   qvel[0] (rad/s)")
for step in range(1, 501):  # 500 steps x 2 ms = 1 second
    mujoco.mj_step(model, data)
    if step in (5, 25, 50, 100, 200, 500):
        print(f"{data.time:5.2f}s   {data.qpos[0]:8.3f}       {data.qvel[0]:8.3f}")

mujoco.mj_forward(model, data)
print("\nend    qpos =", data.qpos, " qvel =", data.qvel)
print("       gripperframe xyz =", data.site("gripperframe").xpos)
