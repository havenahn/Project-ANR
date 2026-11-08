#!/usr/bin/env python
# Look for # line gets changed

import os
import torch
import math
import copy
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



class NeuralNetwork(torch.nn.Module):
    def __init__(self, N_x = 1, N_h = 16, N_e = 10):
        super(NeuralNetwork, self).__init__()
        D_x = N_x
        D_h = N_h
        D_e = N_e
        # Store dimensions in instance variables
        self.D_x = D_x
        self.D_e = D_e
        D_xi = self.D_x + self.D_e

        self.enc = nn.Sequential(
            nn.Linear(D_x, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_e),
        )
        # Linear dynamic model matrices
        self.A = torch.nn.Linear(D_xi, D_xi, bias=False)

    def forward(self, x: torch.Tensor):
        '''Function to propagate the state and encoded state forward in time.
        
        Args:
            x (torch.Tensor): State of the system.
            
        Returns:
            x_tp1 (torch.Tensor): State of the system at the next time step.
            eta_tp1 (torch.Tensor): Encoded state of the system at the next time step.'''
        xs   = x
        eta  = self.enc(xs)
        xi = torch.cat((xs,eta), 1) # lines change
        #xi = torch.cat((xs,eta), 0) # lines change

        x_tp1, eta_tp1 = self.ldm(xi)

        return x_tp1, eta_tp1

    def ldm(self, xi: torch.Tensor):
        '''Function to propagate the encoded state forward in time.
        
        Args:
            eta (torch.Tensor): Encoded state of the system.
            
        Returns:
            eta_tp1 (torch.Tensor): Encoded state of the system at the next time step.'''
        xi_tp1 = self.A(xi)
        #print('xi_tp1: ', xi_tp1.shape)
        eta_tp1 = xi_tp1[:,self.D_x:]
        x_tp1 = xi_tp1[:,:self.D_x]
        return x_tp1, eta_tp1
