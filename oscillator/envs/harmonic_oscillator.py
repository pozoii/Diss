import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class HarmonicOscillatorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, target=0.0):
        super().__init__()

        self.target = target

        self.max_steps = 500
        self.step_count = 0

        self.position_stability = 0.1
        self.velocity_stability = 0.1

        self.stable_steps_required = 5
        self.stable_steps = 0

        # -------------------------
        # MUJOCO MODEL
        # -------------------------
        self.model = mujoco.MjModel.from_xml_path("oscillator/oscillator.xml")
        self.data = mujoco.MjData(self.model)

        # -------------------------
        # SPACES (like GridWorld)
        # -------------------------

        # action = force a_t
        self.action_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(1,),
            dtype=np.float32
        )

        # observation = [x, x_dot, xddot]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

        # -------------------------
        # RENDERING
        # -------------------------
        self.render_mode = render_mode
        self.viewer = None

    # =====================================================
    # OBS + INFO (same role as GridWorld)
    # =====================================================

    def _get_obs(self):
        return np.array([
            self.data.qpos[0],
            self.data.qvel[0],
            self.data.qacc[0]
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "position": self.data.qpos[0],
            "velocity": self.data.qvel[0],
            "acceleration": self.data.qacc[0],
            "target": self.target
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

        if ini is not None:
            self.data.qpos[0] = ini.get("pos",0.0)
            self.data.qvel[0] = ini.get("vel",0.0)
        else:
            # random initial condition (like random agent position)
            self.data.qpos[0] = self.np_random.uniform(-1.0, 1.0)
            self.data.qvel[0] = self.np_random.uniform(-0.5, 0.5)
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

    
        info = self._get_info()

        return obs, info

    # =====================================================
    # STEP (this replaces GridWorld logic)
    # =====================================================

    def step(self, action):
        action = np.clip(action, -50, 50)

        # apply control
        self.data.ctrl[0] = action

        # physics step (THIS replaces your grid transition logic)
        mujoco.mj_step(self.model, self.data)

        info = self._get_info()
        obs = self._get_obs()
        
        x,xdot, xdotdot = obs
        self.step_count += 1

        # reward
        reward = -((x - self.target)**2+ 0.01 * action[0]**2)

        if (abs(x - self.target) < self.position_stability and abs(xdot) < self.velocity_stability):
            self.stable_steps += 1
            
        else:
            self.stable_steps = 0
            
        

        terminated = (self.stable_steps >= self.stable_steps_required)
        truncated = (self.step_count >= self.max_steps)
        

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