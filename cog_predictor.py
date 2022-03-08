from secrets import choice
from configparser import ConfigParser
from dataclasses import dataclass
import sde_lib
import sampling
import util.utils as utils

import models.utils as mutils
import torch
from torchvision.utils import make_grid
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import gc
import cog


@dataclass
class SamplerConfig:
    # Sampling
    sampling_method: str = 'sscs' # choices=['ode', 'em', 'sscs'],
    sampling_solver: str = 'scipy_solver'
    sampling_solver_options: Dict[str, str] = {'solver': 'RK45'}
    sampling_rtol: float = 1e-5
    sampling_atol: float = 1e-5
    sscs_num_stab: float = 0.0
    denoising: bool = True
    n_discrete_steps: int = 500
    striding: str = 'linear' # choices=['linear', 'quadratic', 'logarithmic']
    sampling_eps: float = 1e-5 # TODO: What is this?
    sscs_num_stab: float = 1e-9


def get_model_and_config(model: str, device: torch.Device):
    parser = ConfigParser()
    if model == 'cifar10':
        cc = 'configs/default_cifar10.txt'
        sc = 'configs/specific_cifar10.txt'
        ckpt_path = 'checkpoints/cifar10_800000.pth'
    elif model == 'celeba_hq_256':
        cc = 'configs/default_celeba_paper.txt'
        sc = 'configs/specific_celeba_paper.txt'
        ckpt_path = 'checkpoints/celebahq256_600000.pth'
    config = parser.read([cc, sc])
    config.set('checkpoint', ckpt_path)
    beta_fn = utils.build_beta_fn(config)
    beta_int_fn = utils.build_beta_int_fn(config)
    sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    #@title Setting up the score model
    score_model = mutils.create_model(config).to(device)
    score_model.eval()
    return sde, score_model, config


class CLDSampler(cog.BasePredictor):

    sampling_methods = {
        'Symmetric Splitting CLD Sampler': 'sscs',
        'Euler Maruyama': 'em'
    }

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        assert torch.cuda.is_available(), 'No GPU found, you really want a GPU to run this.'
        self.device = torch.device('cuda')

    @staticmethod
    def save_samples(filepath, x, model_config):
        inverse_scaler = utils.get_data_inverse_scaler(model_config)
        nrow = int(np.sqrt(x.shape[0]))
        image_grid = make_grid(inverse_scaler(x).clamp(0., 1.), nrow)
        plt.axis('off')
        plt.imsave(filepath, image_grid.permute(1, 2, 0).cpu())
        
    def sample(
        self,
        model_name: str = cog.Input(choices=['cifar10', 'celeba_hq_256']),
        n_samples: int = cog.Input(default=16, ge=1, le=64),
        sampler: str = cog.Input(choices=['Symmetric Splitting CLD Sampler', 'Euler Maruyama']),
        n_sampling_steps: int = cog.Input(description='Number of steps to sample for', default=500, ge=1, le=750),
        time_step_striding_type: str = cog.Input(choices=['linear', 'quadratic', 'logarithmic'])
    ) -> cog.Path:
        """Run a single prediction on the model"""
        sde, score_model, model_config = get_model_and_config(model_name, self.device)
        sampling_config = SamplerConfig()
        sampling_config.sampling_method = self.sampling_methods[sampler]
        sampling_config.n_discrete_steps = n_sampling_steps
        sampling_config.striding = time_step_striding_type
        sampling_shape = (n_samples, 3, 32, 32)
        sampler = sampling.get_sampling_fn(sampling_config, sde, sampling_shape, sampling_config.sampling_eps)
        x, _, _ = sampler(score_model)
        fp = './samples.png'
        plt.figure(figsize=(8, 8))
        self.save_samples(fp, x, model_config)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cog.Path(fp)