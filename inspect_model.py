import mujoco

from src import MODEL_PATH

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# MjData starts with xpos all zeros; forward kinematics fills in world positions
mujoco.mj_forward(model, data)

print(f"Bodies ({model.nbody}):")
for i in range(model.nbody):
    parent = model.body(model.body_parentid[i]).name
    print(f"  {i}: {model.body(i).name:<22} parent={parent:<12} xpos={data.xpos[i].round(3)}")

print(f"\nSites ({model.nsite}):")
for i in range(model.nsite):
    body = model.body(model.site_bodyid[i]).name
    print(f"  {i}: {model.site(i).name:<22} on body={body:<12} xpos={data.site_xpos[i].round(3)}")

print(f"\nnq={model.nq}  nv={model.nv}  nu={model.nu}")
print(f"Joints ({model.njnt}):")
for i in range(model.njnt):
    j = model.joint(i)
    print(f"  {i}: {j.name:<14} type={mujoco.mjtJoint(j.type[0]).name:<12} "
          f"qposadr={j.qposadr[0]}  dofadr={j.dofadr[0]}  range={j.range.round(3)}")
