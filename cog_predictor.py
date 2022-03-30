from secrets import choice
from configparser import ConfigParser
from dataclasses import dataclass
import sde_lib
import sampling
import util.utils as utils

import models.utils as mutils
import torch
from torchvision.utils import make_grid
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import gc
import cog



@dataclass
class ModelConfig:
    dataset: str = "cifar10"
    is_image: bool = True
    image_size: int = 32
    center_image: bool = True
    image_channels: int = 3
    # Training
    snapshot_freq: int = 10000
    snapshot_threshold: int = 1
    log_freq: int = 5000
    eval_freq: int = 20000
    likelihood_threshold: int = 2000000
    likelihood_freq: int = 50000
    fid_freq: int = 50000
    fid_threshold: int = 100000
    fid_samples_training: int = 20000
    n_eval_batches: int = 1
    n_likelihood_batches: int = 1
    n_warmup_iters: int = 100000
    n_train_iters: int = 800000
    save_freq: int  = 50000
    save_threshold: int = 300000
    # Autocast
    autocast_train: bool = True
    autocast_eval: bool = True
    # Sampling
    sampling_method: str = "ode"
    sampling_eps: float = 1e-3
    denoising: bool = True
    name                = "ncsnpp"
    normalization       = "GroupNorm"
    nonlinearity        = "swish"
    n_channels          = 128
    ch_mult             = (1, 2, 2, 2)
    attn_resolutions    = 16
    resamp_with_conv    = True
    use_fir             = True
    fir_kernel          = (1, 3, 3, 1)
    skip_rescale        = True
    resblock_type       = "biggan"
    progressive         = None
    progressive_input   = "residual"
    progressive_combine = "sum"
    attention_type      = "ddpm"
    init_scale          = 0.0
    fourier_scale       = 16
    conv_size           = 3
    embedding_type      = "fourier"
    mixed_score         = True
    n_resblocks         = 8
    ema_rate            = 0.9999
    numerical_eps       = 1e-9
    sde                 = "cld"
    beta_type           = "linear"
    beta0               = 4.0
    beta1               = 0.0
    m_inv               = 4.0
    gamma               = 0.04
    # Optimization
    optimizer           = "Adam"
    learning_rate       = 2e-4
    grad_clip           = 1.0
    dropout             = 0.1
    weight_decay        = 0.0
    # Objective
    cld_objective       = "hsm"
    loss_eps            = 1e-5
    weighting           = "reweightedv2"


@dataclass
class SamplerConfig:
    # Sampling
    sampling_method: str = 'sscs' # choices=['ode', 'em', 'sscs'],
    sampling_solver: str = 'scipy_solver'
    sampling_solver_options: Optional[Dict[str, str]] = None
    sampling_rtol: float = 1e-5
    sampling_atol: float = 1e-5
    sscs_num_stab: float = 0.0
    denoising: bool = True
    n_discrete_steps: int = 500
    striding: str = 'linear' # choices=['linear', 'quadratic', 'logarithmic']
    sampling_eps: float = 1e-5 # TODO: What is this?
    sscs_num_stab: float = 1e-9

    def __post_init__(self):
        if self.sampling_solver_options is None:
            self.sampling_solver_options = {'solver': 'RK45'}


def get_model_and_config(model: str, device: str):
    config = ConfigParser()
    if model == 'cifar10':
        cc = 'configs/default_cifar10.txt'
        sc = 'configs/specific_cifar10.txt'
        ckpt_path = 'checkpoints/cifar10_800000.pth'
    elif model == 'celeba_hq_256':
        cc = 'configs/default_celeba_paper.txt'
        sc = 'configs/specific_celeba_paper.txt'
        ckpt_path = 'checkpoints/celebahq256_600000.pth'
    config.read([cc, sc])
    config['checkpoint'] = ckpt_path
    model_config = ModelConfig(**{k: v for k, v in config.items('default')})
    beta_fn = utils.build_beta_fn(model_config)
    beta_int_fn = utils.build_beta_int_fn(model_config)
    sde = sde_lib.CLD(model_config, beta_fn, beta_int_fn)
    #@title Setting up the score model
    score_model = mutils.create_model(model_config).to(device)
    score_model.eval()
    return sde, score_model, model_config


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
        
    def predict(
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