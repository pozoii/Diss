import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("ball_on_plate/ball_on_plate.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep


# =====================================================
# Helper functions
# =====================================================

def reset():
    mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def log_state(tag):
    print(
        f"{tag} | "
        f"roll={data.qpos[0]:.4f}, pitch={data.qpos[1]:.4f} | "
        f"roll_vel={data.qvel[0]:.4f}, pitch_vel={data.qvel[1]:.4f}"
    )


# =====================================================
# TEST PHASES
# =====================================================

phases = [
    "PASSIVE_STABILITY",
    "ROLL_STEP",
    "PITCH_STEP",
    "ZERO_INPUT_CHECK",
    "OSCILLATION_DAMPING_CHECK",
]


phase = 0
step_counter = 0


with mujoco.viewer.launch_passive(model, data) as viewer:

    reset()

    while viewer.is_running():

        name = phases[phase]

        # =================================================
        # 1. PASSIVE STABILITY (no torque)
        # =================================================
        if name == "PASSIVE_STABILITY":
            data.ctrl[:] = 0.0

            if step_counter % 200 == 0:
                log_state("PASSIVE")

            if step_counter > 1000:
                phase += 1
                step_counter = 0
                reset()

        # =================================================
        # 2. ROLL STEP RESPONSE
        # =================================================
        elif name == "ROLL_STEP":
            data.ctrl[0] = 0.3
            data.ctrl[1] = 0.0

            if step_counter % 200 == 0:
                log_state("ROLL STEP")

            if step_counter > 1000:
                phase += 1
                step_counter = 0
                reset()

        # =================================================
        # 3. PITCH STEP RESPONSE
        # =================================================
        elif name == "PITCH_STEP":
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.3

            if step_counter % 200 == 0:
                log_state("PITCH STEP")

            if step_counter > 1000:
                phase += 1
                step_counter = 0
                reset()

        # =================================================
        # 4. ZERO INPUT CHECK (should settle)
        # =================================================
        elif name == "ZERO_INPUT_CHECK":
            data.ctrl[:] = 0.0

            if step_counter % 200 == 0:
                log_state("ZERO INPUT")

            if step_counter > 1000:
                phase += 1
                step_counter = 0
                reset()

        # =================================================
        # 5. OSCILLATION / DAMPING TEST
        # =================================================
        elif name == "OSCILLATION_DAMPING_CHECK":
            if step_counter < 2:
                data.ctrl[:] = [10, -10]
            else:
                data.ctrl[:] = [0.0, 0.0]

            if step_counter % 100 == 0:
                log_state("OSC")

            if step_counter > 1000:
                phase = 0
                step_counter = 0
                reset()

        # =================================================
        # STEP SIMULATION
        # =================================================
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        step_counter += 1