import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("ball_on_plate/ball_on_plate.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep


# =====================================================
# RESET (IMPORTANT: do NOT wipe qpos/qvel manually)
# =====================================================
def reset():
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


# =====================================================
# LOGGING
# =====================================================
def log(tag):
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_pos = data.geom_xpos[ball_id]

    roll = data.qpos[0]
    pitch = data.qpos[1]

    print(
        f"{tag} | "
        f"plate(r,p)=({roll:.3f},{pitch:.3f}) | "
        f"ball(x,y,z)=({ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f})"
    )


# =====================================================
# TEST PHASES
# =====================================================
phases = [
    "GRAVITY_FALL",
    "CONTACT_CHECK",
    "ROLL_ONLY",
    "PITCH_ONLY",
    "FRICTION_TEST",
    "ENERGY_TEST",
    "RECOVERY_TEST",
]
phases = [
    "ROLL_POS",
    "ROLL_NEG",
    "PITCH_POS",
    "PITCH_NEG",
]
phase = 0
step = 0


with mujoco.viewer.launch_passive(model, data) as viewer:

    reset()

    while viewer.is_running():

        name = phases[phase]

        # =================================================
        # 1. GRAVITY ONLY (plate locked)
        # =================================================
        if name == "GRAVITY_FALL":
            data.ctrl[:] = 0

            if step % 50 == 0:
                log("GRAVITY")

            if step > 500:
                phase += 1
                step = 0
                reset()

        # =================================================
        # 2. CONTACT CHECK (ball must sit on plate center)
        # =================================================
        elif name == "CONTACT_CHECK":
            data.ctrl[:] = 0

            if step % 50 == 0:
                log("CONTACT")

            if step > 500:
                phase += 1
                step = 0
                reset()

        # =================================================
        # 3. ROLL ONLY (pure x-direction response)
        # =================================================
        elif name == "ROLL_ONLY":
            data.ctrl[0] = 0.5
            data.ctrl[1] = 0.0

            if step % 50 == 0:
                log("ROLL")

            if step > 800:
                phase += 1
                step = 0
                reset()

        # =================================================
        # 4. PITCH ONLY (pure y-direction response)
        # =================================================
        elif name == "PITCH_ONLY":
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.5

            if step % 50 == 0:
                log("PITCH")

            if step > 800:
                phase += 1
                step = 0
                reset()

        # =================================================
        # 5. FRICTION TEST (ball should slow, not jitter)
        # =================================================
        elif name == "FRICTION_TEST":
            data.ctrl[:] = 0

            if step == 0:
                data.qvel[2:5] += np.array([0.5, 0.2, 0])  # small kick

            if step % 50 == 0:
                log("FRICTION")

            if step > 800:
                phase += 1
                step = 0
                reset()

        # =================================================
        # 6. ENERGY TEST (constant oscillation)
        # =================================================
        elif name == "ENERGY_TEST":
            data.ctrl[:] = 0.3 * np.sin(step * 0.05)

            if step % 50 == 0:
                log("ENERGY")

            if step > 800:
                phase += 1
                step = 0
                reset()

        # =================================================
        # 7. RECOVERY TEST (disturb then release)
        # =================================================
        elif name == "RECOVERY_TEST":

            if step < 100:
                data.ctrl[:] = [1.0, -1.0]
            else:
                data.ctrl[:] = [0.0, 0.0]

            if step % 50 == 0:
                log("RECOVERY")

            if step > 800:
                phase = 0
                step = 0
                reset()

        elif name == "ROLL_POS":
            ANGLE = np.deg2rad(15)
            KP = 2
            KD = 0.5
            desired_roll = ANGLE
            desired_pitch = 0.0

            roll = data.qpos[0]
            pitch = data.qpos[1]

            roll_dot = data.qvel[0]
            pitch_dot = data.qvel[1]

            data.ctrl[0] = KP*(desired_roll-roll) - KD*roll_dot
            data.ctrl[1] = KP*(desired_pitch-pitch) - KD*pitch_dot

            if step % 50 == 0:
                log("ROLL +15°")

            if step > 500:
                phase += 1
                step = 0
                reset()

        elif name == "ROLL_NEG":
            ANGLE = np.deg2rad(15)
            KP = 2
            KD = 0.5
            desired_roll = -ANGLE
            desired_pitch = 0.0

            roll = data.qpos[0]
            pitch = data.qpos[1]

            roll_dot = data.qvel[0]
            pitch_dot = data.qvel[1]

            data.ctrl[0] = KP*(desired_roll-roll) - KD*roll_dot
            data.ctrl[1] = KP*(desired_pitch-pitch) - KD*pitch_dot

            if step % 50 == 0:
                log("ROLL -15°")

            if step > 500:
                phase += 1
                step = 0
                reset()
        
        elif name == "PITCH_POS":
            ANGLE = np.deg2rad(15)
            KP = 2
            KD = 0.5
            desired_roll = 0.0
            desired_pitch = ANGLE

            roll = data.qpos[0]
            pitch = data.qpos[1]

            roll_dot = data.qvel[0]
            pitch_dot = data.qvel[1]

            data.ctrl[0] = KP*(desired_roll-roll) - KD*roll_dot
            data.ctrl[1] = KP*(desired_pitch-pitch) - KD*pitch_dot

            if step % 50 == 0:
                log("PITCH +15°")

            if step > 500:
                phase += 1
                step = 0
                reset()

        elif name == "PITCH_NEG":
            ANGLE = np.deg2rad(15)
            KP = 2
            KD = 0.5
            desired_roll = 0.0
            desired_pitch = -ANGLE

            roll = data.qpos[0]
            pitch = data.qpos[1]

            roll_dot = data.qvel[0]
            pitch_dot = data.qvel[1]

            data.ctrl[0] = KP*(desired_roll-roll) - KD*roll_dot
            data.ctrl[1] = KP*(desired_pitch-pitch) - KD*pitch_dot

            if step % 50 == 0:
                log("PITCH -15°")

            if step > 500:
                phase += 1
                step = 0
                reset()

        # STEP SIM
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        step += 1

    