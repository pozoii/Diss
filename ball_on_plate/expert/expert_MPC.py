import numpy as np
from ball_on_plate.env.ball_on_plate import BallOnPlateEnv
import pandas as pd
import mujoco
from joblib import load
import time

class ExpertMPC:
    def __init__(self, env, plate_models, H=10, max_torque=10, plate_joint_limit=np.deg2rad(30), max_iters=10, x_target = np.zeros(8)):

        self.H = H  # Horizon length
        #self.dt= env.dt
        self.dt= 0.02
        self.plate_models = plate_models
        self.max_torque = max_torque
        self.plate_joint_limit = plate_joint_limit
        self.max_iters = max_iters
        self.x_target = x_target

        # Q, R from provided cost function parameters

        self.Q = np.diag([25,25, 4, 4, 1,1,0,0])
        self.R = np.diag([0.1,0.1])


        # Warm start buffer
        self.U_guess = np.zeros((self.H, 2))
    
    def predict_next_state(self, state, action):

        dt = self.dt
        
        
        x,y,xd,yd,alpha,beta,alphad,betad = state

        tau_alpha, tau_beta = action

        # plate dynamics

        roll_model , pitch_model = self.plate_models

        alphadd = (roll_model.intercept_ + roll_model.coef_[0] * tau_alpha + roll_model.coef_[1] * alpha + roll_model.coef_[2] * alphad)

        betadd = (pitch_model.intercept_ + pitch_model.coef_[0] * tau_beta + pitch_model.coef_[1] * beta + pitch_model.coef_[2] * betad)

        # ball dynamics
        g = 9.81
        C = 5/7*g

        xdd = -C *alpha
        ydd = -C *beta

        # Euler integration
        xd_next = xd + dt * xdd
        yd_next = yd + dt * ydd

        x_next = x + dt * xd_next
        y_next = y + dt * yd_next

        alphad_next = alphad + dt * alphadd
        betad_next  = betad  + dt * betadd

        alpha_next = alpha + dt * alphad_next
        beta_next  = beta  + dt * betad_next

        # Respect plate joint limits
        plate_joint_limit = self.plate_joint_limit
        alpha_next = np.clip(alpha_next, -plate_joint_limit, plate_joint_limit)
        beta_next = np.clip(beta_next, -plate_joint_limit, plate_joint_limit) 

        alphad_next = np.where(((alpha_next >= plate_joint_limit) & (alphad_next > 0)) | ((alpha_next <= -plate_joint_limit) & (alphad_next < 0)), 0.0, alphad_next,)
        betad_next  = np.where(((beta_next  >= plate_joint_limit) & (betad_next  > 0)) | ((beta_next  <= -plate_joint_limit) & (betad_next  < 0)), 0.0, betad_next,)

        return np.array([x_next, y_next, xd_next, yd_next, alpha_next, beta_next, alphad_next, betad_next])

    def get_jacobians(self):
        """
        Analytic Jacobians of predict_next_state().
        Clipping is ignored.
        """

        roll_model , pitch_model = self.plate_models
        dt = self.dt

        C = 5.0 / 7.0 * 9.81
        
        ar = roll_model.coef_
        ap = pitch_model.coef_

        # roll model:
        # alphadd = b + ar[0]*tau + ar[1]*alpha + ar[2]*alphad

        # pitch model:
        # betadd = b + ap[0]*tau + ap[1]*beta + ap[2]*betad

        A = np.eye(8)
        # Ball
        # x
        A[0,2] += dt
        A[0,4] += -C * dt**2
        # y
        A[1,3] += dt
        A[1,5] += -C * dt**2
        # xdot
        A[2,4] += -C * dt
        # ydot
        A[3,5] += -C * dt

        # Plate
        # alpha
        A[4,4] += ar[1] * dt**2
        A[4,6] += dt + ar[2] * dt**2
        # beta
        A[5,5] += ap[1] * dt**2
        A[5,7] += dt + ap[2] * dt**2
        # alphadot
        A[6,4] += ar[1] * dt
        A[6,6] += ar[2] * dt
        # betadot
        A[7,5] += ap[1] * dt
        A[7,7] += ap[2] * dt

        # --------------------------------------------------
        # Action Jacobian
        # --------------------------------------------------

        B = np.zeros((8,2))

        # roll torque
        B[4,0] = ar[0] * dt**2
        B[6,0] = ar[0] * dt
        # pitch torque
        B[5,1] = ap[0] * dt**2
        B[7,1] = ap[0] * dt

        return A, B

    def solve_ilqr(self, x0, U_init):
        """The iLQR Solver"""
        U = U_init.copy()
        X = np.zeros((self.H + 1, 8))
        X[0] = x0
        
        # Initial Rollout
        for t in range(self.H):
            X[t+1] = self.predict_next_state(X[t], U[t])
            
        for _ in range(self.max_iters):
            # Backward Pass
            ks = [np.zeros(2) for _ in range(self.H)]
            Ks = [np.zeros((2, 8)) for _ in range(self.H)] 
            
            # Terminal Value Function derivatives
            Vx = self.Q @ (X[-1]-self.x_target)
            Vxx = self.Q

            
            for t in reversed(range(self.H)):
                A, B = self.get_jacobians()

                # Gradients of the cost
                lx = 2*self.Q @ (X[t]-self.x_target)
                lu = 2*self.R @ U[t]
                lxx =  2*self.Q
                luu =  2*self.R
                lux = np.zeros((2, 8))

                ### FILL IN HERE ### hint: Q-function derivatives, control gains, value function update
                Qx= lx + np.transpose(A) @ Vx
                Qu= lu + np.transpose(B) @ Vx
                Qxx= lxx + np.transpose(A) @ Vxx @ A
                Quu= luu + np.transpose(B) @ Vxx @ B 
                Qux= lux + np.transpose(B) @ Vxx @ A

                #control gains
                reg = 1e-6
                Quu_reg = Quu + reg*np.eye(2)
                ks[t] = - np.linalg.inv(Quu_reg) @ Qu
                Ks[t] = - np.linalg.inv(Quu_reg) @ Qux

                #value function update
                Vx = Qx - np.transpose(Qux) @ np.linalg.inv(Quu) @ Qu
                Vxx = Qxx - np.transpose(Qux) @ np.linalg.inv(Quu) @ Qux
                
            # Forward Pass (Line search simplified for brevity)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            U_new = np.zeros_like(U)

            alphas = [1, 0.5, 0.25, 0.1, 0.05]
            best = np.inf 
            
            for alpha in alphas:
                X_new[0] = x0
                J = 0
                for t in (range(self.H)):
                    U_new[t] = np.clip(U[t] + alpha * ks[t] + Ks[t] @ (X_new[t] - X[t]), -self.max_torque, self.max_torque)
                    X_new[t+1] = self.predict_next_state(X_new[t], U_new[t])
                    dx = X_new[t] - self.x_target
                    J += dx.T @ self.Q @ dx + U_new[t].T @ self.R @ U_new[t]

                dx = X_new[self.H] - self.x_target
                J += dx.T @ self.Q @ dx
    
                if J < best:
                    best = J
                    best_X = X_new.copy()
                    best_U = U_new.copy()

            X, U = best_X, best_U

        ks = np.array(ks)
        Ks = np.array(Ks)
        return U, X, ks, Ks

    def reset(self):
        """Reset the warm start buffer for a new episode"""
        self.U_guess = np.random.uniform(-1,1,(self.H,2))
    
    def control(self, state):
        """MPC interface: solve and shift"""
        U_opt, _, _, _ = self.solve_ilqr(state, self.U_guess)
        
        # Extract first action and ensure it's a scalar
        action = U_opt[0]
        
        action = np.clip(action,-self.max_torque,self.max_torque)
        
        # Warm start shift
        self.U_guess[:-1] = U_opt[1:]
        self.U_guess[-1] = 0
        
        return action

def run_ExpertMPC(episodes=1000,max_steps=5000, H=30, sim = False):

        env = BallOnPlateEnv(render_mode= None)

        models = load("ball_on_plate/dynamics/plate_models.joblib")

        plate_models = (models["roll_model"],models["pitch_model"],)

        expert = ExpertMPC(env=env,plate_models=plate_models, H=H)

        records =[]

        for ep in range(episodes):
            print(f'ep:{ep}')

            qpos_history=[]
            qvel_history=[]
            actions=[]

            obs, info = env.reset()

            expert.reset()

            for step in range(max_steps):
                if step % 100 ==0:
                    print(f'step:{step}')

                action = expert.control(obs)


                next_obs, reward, terminated, truncated, info = env.step(action)

                qpos_history.append(env.data.qpos.copy())
                qvel_history.append(env.data.qvel.copy())
                actions.append(action.copy())
                #env.render()

                records.append({

                    "episode": ep,
                    "step": step,

                    # Current state
                    "x": obs[0],
                    "y": obs[1],
                    "xdot": obs[2],
                    "ydot": obs[3],
                    "alpha": obs[4],
                    "beta": obs[5],
                    "alphadot": obs[6],
                    "betadot": obs[7],

                    # Action
                    "roll_torque": action[0],
                    "pitch_torque": action[1],

                    # Next state
                    "x_next": next_obs[0],
                    "y_next": next_obs[1],
                    "xdot_next": next_obs[2],
                    "ydot_next": next_obs[3],
                    "alpha_next": next_obs[4],
                    "beta_next": next_obs[5],
                    "alphadot_next": next_obs[6],
                    "betadot_next": next_obs[7],

                    # Done flag
                    "done": terminated or truncated or info.get("ball_lost", False),
                })

                obs = next_obs

                if terminated or truncated or info.get("ball_lost", False):
                    break
            if sim:
            
                viewer = mujoco.viewer.launch_passive(env.model, env.data)

                for qpos, qvel in zip(qpos_history, qvel_history):

                    # set simulator state
                    env.data.qpos[:] = qpos
                    env.data.qvel[:] = qvel

                    mujoco.mj_forward(env.model, env.data)

                    viewer.sync()
                    time.sleep(0.005)

            viewer.close()

        env.close()

        return pd.DataFrame(records)



    


#tech mech meca moca mock lock look loop poop
#tech mech mesh mosh posh 



  
        
