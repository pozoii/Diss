import torch
import torch.nn as nn


class HarmonicOscillatorDynamics(nn.Module):

    def __init__(self, dt, m=1.0, k=10.0):
        super().__init__()

        self.dt = dt
        self.m = m
        self.k = k

    def forward(self, state, action):

        x = state[:,0]
        xdot = state[:,1]

        u = action.squeeze(-1)

        xddot = (u - self.k*x)/self.m

        xdot_next = xdot + self.dt*xddot
        x_next = x + self.dt*xdot_next

        
        next_state = torch.stack([x_next,xdot_next,],dim=1)

        return next_state