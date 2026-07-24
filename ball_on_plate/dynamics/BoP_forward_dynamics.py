import torch
import torch.nn as nn

class BallOnPlateDynamics(nn.Module):

    def __init__(self,dt,roll_model,pitch_model,plate_joint_limit):
        super().__init__()

        self.register_buffer("dt",torch.tensor(dt, dtype=torch.float32))
        self.register_buffer("plate_joint_limit",torch.tensor(plate_joint_limit, dtype=torch.float32))
        self.plate_joint_limit = plate_joint_limit

        self.register_buffer("roll_coef",torch.tensor(roll_model.coef_, dtype=torch.float32))
        self.register_buffer("roll_intercept",torch.tensor(roll_model.intercept_, dtype=torch.float32))
        self.register_buffer("pitch_coef",torch.tensor(pitch_model.coef_, dtype=torch.float32))
        self.register_buffer("pitch_intercept",torch.tensor(pitch_model.intercept_, dtype=torch.float32))

        self.register_buffer("C",torch.tensor(5 * 9.81 / 7, dtype=torch.float32))

      

    def forward(self, state, action):

        dt = self.dt
        
        
        x = state[:, 0]
        y = state[:, 1]
        xd = state[:, 2]
        yd = state[:, 3]
        alpha = state[:, 4]
        beta = state[:, 5]
        alphad = state[:, 6]
        betad = state[:, 7]

        tau_alpha = action[:,0]
        tau_beta = action [:,1]

        # plate dynamics

        alphadd = (self.roll_intercept + self.roll_coef[0] * tau_alpha + self.roll_coef[1] * alpha + self.roll_coef[2] * alphad)

        betadd = (self.pitch_intercept + self.pitch_coef[0] * tau_beta + self.pitch_coef[1] * beta + self.pitch_coef[2] * betad)

        xdd = -self.C *alpha
        ydd = -self.C *beta

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
        alpha_next = torch.clamp(alpha_next, -plate_joint_limit, plate_joint_limit)
        beta_next = torch.clamp(beta_next, -plate_joint_limit, plate_joint_limit) 

        alphad_next = torch.where(((alpha_next >= plate_joint_limit) & (alphad_next > 0)) | ((alpha_next <= -plate_joint_limit) & (alphad_next < 0)), torch.zeros_like(alphad_next), alphad_next,)
        betad_next  = torch.where(((beta_next  >= plate_joint_limit) & (betad_next  > 0)) | ((beta_next  <= -plate_joint_limit) & (betad_next  < 0)), torch.zeros_like(betad_next), betad_next,)

        return torch.stack([x_next, y_next, xd_next, yd_next, alpha_next, beta_next, alphad_next, betad_next], dim=1)
