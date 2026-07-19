import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("ball_on_plate/ball_on_plate.xml")
data = mujoco.MjData(model)

print("\n==============================")
print("JOINT STRUCTURE (GROUND TRUTH)")
print("==============================\n")

for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    qposadr = model.jnt_qposadr[j]
    dofadr = model.jnt_dofadr[j]
    jtype = model.jnt_type[j]

    print(f"Joint: {name}")
    print(f"  qposadr: {qposadr}")
    print(f"  dofadr : {dofadr}")
    print(f"  type   : {jtype}")
    print()


print("\n==============================")
print("QPOS / QVEL SHAPES")
print("==============================\n")

print("nq  =", model.nq)
print("nv  =", model.nv)
print("qpos =", data.qpos)
print("qvel =", data.qvel)


print("\n==============================")
print("FREE JOINT DETECTION CHECK")
print("==============================\n")

for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print("FREE JOINT FOUND:", name)
        print("  qposadr:", model.jnt_qposadr[i])
        print("  dofadr :", model.jnt_dofadr[i])


print("\n==============================")
print("STATE RESPONSE TEST")
print("==============================\n")

def reset():
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

reset()

# --- baseline
print("BASELINE")
print("qpos:", data.qpos)
print("qvel:", data.qvel)

# --- roll excitation
data.ctrl[:] = [0.5, 0.0]
for _ in range(50):
    mujoco.mj_step(model, data)

print("\nAFTER ROLL INPUT")
print("qpos:", data.qpos)
print("qvel:", data.qvel)

# --- pitch excitation
reset()
data.ctrl[:] = [0.0, 0.5]
for _ in range(50):
    mujoco.mj_step(model, data)

print("\nAFTER PITCH INPUT")
print("qpos:", data.qpos)
print("qvel:", data.qvel)


print("\n==============================")
print("BALL STATE EXTRACTION CHECK")
print("==============================\n")

ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

print("Ball geom id:", ball_geom)
print("Ball position (geom_xpos):", data.geom_xpos[ball_geom])

ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
print("Ball body id:", ball_body)

print("Ball full body velocity (cvel):", data.cvel[ball_body])


print("\n==============================")
print("QVEL vs PHYSICAL VELOCITY CONSISTENCY")
print("==============================\n")

print("qvel:", data.qvel)
print("cvel (ball):", data.cvel[ball_body])