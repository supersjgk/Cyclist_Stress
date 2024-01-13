import torch
import torch.nn as nn
import gpytorch
import numpy as np
import time

class WGP(gpytorch.models.ExactGP):
    """
        Implementation of Multi-task Gaussian Process using GPyTorch: https://github.com/cornellius-gp/gpytorch
        Multitask GP regression learns similarities in the tasks simultaneously.
        For num_tasks = 100, check NOTE in wgp_pipeline.py
    """
    def __init__(self, train_x=None, train_y=None, likelihood=None):
        super(WGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=100
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=100
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class WarpClass(nn.Module):
    """
        Implementation of Multi-task Warped Gaussian Process.
        Likelihood is defined on z and not on y.
        z = g(y) = a * tanh(b * y + c) + d, where g is the warping function and should be monotonic and real.
        After training, the inverse of warping function is applied to the predictions to get final outputs.
        This warping introduces a new set of trainable parameters (a,b,c,d) and a loss term defined by the loss() function below
    """
    def __init__(self):
        super(WarpClass,self).__init__()
        # These initial values of the parameters gave the best results
        self.a = nn.Parameter(torch.tensor(12.4))
        self.b = nn.Parameter(torch.tensor(10.5))
        self.c = nn.Parameter(torch.tensor(10.5))
        self.d = nn.Parameter(torch.tensor(0.5))

    def forward(self,y):
        cloned_a = self.a.clone().detach()
        cloned_b = self.b.clone().detach()
        cloned_c = self.c.clone().detach()
        cloned_d = self.d.clone().detach()
        z = cloned_a * torch.tanh(cloned_b * y + cloned_c) + cloned_d  # g(y)
        return z

    def grad(self, y):
        b_y_c = self.b * y + self.c
        cosh_result = torch.cosh(b_y_c)
        squared_cosh = cosh_result ** 2
        der_i = self.a * self.b / squared_cosh
        return der_i

    def inverse(self,z):
        inv = (torch.atanh((z-self.d)/self.a)-self.c)/self.b # g^-1(y)
        return inv

    def loss(self, y):
        sum = torch.tensor(0.0, device=y.device)
        for i in range(y.shape[1]):
            y_ij = torch.tensor(0.0, device=y.device)
            for j in range(y.shape[0]):
                y_ij += torch.log(self.grad(y[j][i]))
            sum += y_ij
        return sum/y.numel()

    def inv_loss(self,x,y,model,likelihood):
        pred = likelihood(model(x)).mean
        return torch.mean(torch.square(pred-y))
