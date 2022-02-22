from pytorch_diffusion import Diffusion
from Cifar10_models import ResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F

class With_Diffusion(nn.Module):
    def __init__(self, diffusion_model, model, steps, with_grad=True):
        super(With_Diffusion, self).__init__()
        self.diffusion = diffusion_model
        self.model = model
        self.steps = steps
        self.with_grad = with_grad

    def forward(self, x):
        steps = self.steps
        x = x*2 -1
        samples_diff = self.diffusion.front(x.shape[0], x=x, n_steps=steps)
        samples_dn = self.diffusion.back(x.shape[0], x=samples_diff, curr_step=steps, with_grad=self.with_grad)
        samples_dn = (samples_dn + 1)*0.5
        out = self.model(samples_dn)
        return out

def model_with_diffusion(diffusion_model, model, steps, with_grad=True):
    return With_Diffusion(diffusion_model, model, steps, with_grad)