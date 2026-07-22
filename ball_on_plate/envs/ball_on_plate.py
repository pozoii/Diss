import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class BallOnPlateEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, target=np.array([0.0, 0.0], dtype=np.float32), ball=True, max_pos_reset=0.2, max_vel_reset=0.5, stable_pos_thresh=0.02 ,stable_vel_thresh=0.02, settling_window=50):
        
        super().__init__()

        self.target = target
        self.ball = ball

        self.max_steps = 5000
        self.step_count = 0
        self.stable_steps = 0

        self.joints = {}
        self.geoms = {}
        self.prev_vel = np.zeros(2, dtype=np.float32)
        self.finite_diff_acc = np.zeros(2, dtype=np.float32)

        self.max_pos_reset = max_pos_reset
        self.max_vel_reset = max_vel_reset
        self.stable_pos_thresh = stable_pos_thresh
        self.stable_vel_thresh = stable_vel_thresh
        self.settling_window = settling_window


        # -------------------------
        # MUJOCO MODEL
        # -------------------------
        if self.ball==True:
            self.model = mujoco.MjModel.from_xml_path("ball_on_plate/ball_on_plate.xml")
        else:
            self.model = mujoco.MjModel.from_xml_path("ball_on_plate/only_plate.xml")
            
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep


        # Joints
        for joint_id in range(self.model.njnt):

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            self.joints[name] = {
                "id": joint_id,
                "qposadr": self.model.jnt_qposadr[joint_id],
                "qveladr": self.model.jnt_dofadr[joint_id],
                "type": self.model.jnt_type[joint_id],
            }

        # Geometries
        for geom_id in range(self.model.ngeom):

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
                )

            self.geoms[name] = {
                "id": geom_id,
                "size": self.model.geom_size[geom_id].copy(),
                "type": self.model.geom_type[geom_id],
                }
        self.plate_size = self.geoms['plate']['size']
        self.ball_radius = self.geoms['ball_geom']['size'][0]
        self.ball_lost = False
        self.ball_stable = False

        # ----------------
        # SPACES 
        # ----------------

        # action = roll, pitch torques
        self.action_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(2,),
            dtype=np.float32
        )

        # observation:
        # [x, y, x_dot, y_dot, alpha, beta, alpha_dot, beta_dot]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )

        # -------------------------
        # RENDERING
        # -------------------------
        self.render_mode = render_mode
        self.viewer = None

    # =====================================================
    # Check if the ball fell off the plate
    # =====================================================
    def _ball_fell_off(self,obs):

        x, y = obs[0], obs[1]

        x_limit = self.plate_size[0] - self.ball_radius
        y_limit = self.plate_size[1] - self.ball_radius

        return (abs(x) > x_limit or abs(y) > y_limit)
    
    def _ball_stabilised(self,obs):

        target = self.target

        x, y, xdot, ydot = obs[0], obs[1], obs[2], obs[3]
        pos_err = np.linalg.norm(np.array([x,y])-target)
        vel_err = np.linalg.norm(np.array([xdot,ydot]))

        return (pos_err < self.stable_pos_thresh and vel_err < self.stable_vel_thresh)
    
    # =====================================================
    # OBS + INFO (same role as GridWorld)
    # =====================================================

    def _get_obs(self):

        roll = self.data.qpos[self.joints["roll"]["qposadr"]]
        pitch = self.data.qpos[self.joints["pitch"]["qposadr"]]

        roll_dot = self.data.qvel[self.joints["roll"]["qveladr"]]
        pitch_dot = self.data.qvel[self.joints["pitch"]["qveladr"]]

        ball_qpos = self.joints["ball_joint"]["qposadr"]
        ball_qvel = self.joints["ball_joint"]["qveladr"]

        ball_pos = self.data.qpos[ball_qpos : ball_qpos + 3]
        ball_vel = self.data.qvel[ball_qvel : ball_qvel + 3]

        # observation:
        # [x, y, x_dot, y_dot, alpha, beta, alpha_dot, beta_dot]
        return np.array([ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1], roll, pitch, roll_dot, pitch_dot],
                        dtype=np.float32)

    def _get_info(self):
        return {
            "xddot": self.finite_diff_acc[0],
            "yddot": self.finite_diff_acc[1],
            "ball_lost": self.ball_lost,
            "ball_stable": self.ball_stable,
            "ball_stable_concat": self.stable_steps,
            "target": self.target,
            "step": self.step_count,
            }

    # =====================================================
    # RESET (same structure as GridWorld)
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.stable_steps = 0

        ini = None
        if options is not None:
            ini = options.get("ini", None)

        # Joint addresses
        ball_qpos = self.joints["ball_joint"]["qposadr"]
        ball_qvel = self.joints["ball_joint"]["qveladr"]

        if ini is not None:

            # Ball position
            self.data.qpos[ball_qpos] = ini.get("xpos", 0.0)
            self.data.qpos[ball_qpos + 1] = ini.get("ypos", 0.0)

            #We keep z and orientation quaternion unchanged

            # Ball velocity
            self.data.qvel[ball_qvel] = ini.get("xvel", 0.0)
            self.data.qvel[ball_qvel + 1] = ini.get("yvel", 0.0)

            # We keep z velocity unchanged
        else:

            # Random ball position on the plate
            self.data.qpos[ball_qpos] = self.np_random.uniform(-self.max_pos_reset, self.max_pos_reset)
            self.data.qpos[ball_qpos + 1] = self.np_random.uniform(-self.max_pos_reset, self.max_pos_reset)

            # Random ball velocity
            self.data.qvel[ball_qvel] = self.np_random.uniform(-self.max_vel_reset, self.max_vel_reset)
            self.data.qvel[ball_qvel + 1] = self.np_random.uniform(-self.max_vel_reset, self.max_vel_reset)

        
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        self.prev_vel[:] = obs[2:4]
        self.finite_diff_acc[:] = 0.0
        self.ball_lost = False
        self.ball_stable = False

        info = self._get_info()

        return obs, info

    # =====================================================
    # STEP (this replaces GridWorld logic)
    # =====================================================

    def step(self, action):

        self.step_count += 1

        action = np.clip(action, -50, 50)

        # apply control
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]

        ball_qvel = self.joints["ball_joint"]["qveladr"]
        self.prev_vel[0] = self.data.qvel[ball_qvel]
        self.prev_vel[1] = self.data.qvel[ball_qvel + 1]

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        self.ball_lost = self._ball_fell_off(obs)
        self.ball_stable = self._ball_stabilised(obs)

        if self.ball_stable:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        # reward
        reward = 0

        x, y, xdot, ydot = obs[0], obs[1], obs[2], obs[3]
        

        # finite difference acceleration
        self.finite_diff_acc[0] = (xdot - self.prev_vel[0]) / self.dt
        self.finite_diff_acc[1] = (ydot - self.prev_vel[1]) / self.dt


        info = self._get_info()

       
        terminated = self.step_count >= self.max_steps 
        truncated = self.ball_lost or self.stable_steps >= self.settling_window
        

        return obs, reward, terminated, truncated, info

    # =====================================================
    # RENDER (analogous to pygame render)
    # =====================================================

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

        # IMPORTANT: keep event loop alive
        import time
        time.sleep(self.model.opt.timestep)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
