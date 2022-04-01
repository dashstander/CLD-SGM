import dataclasses
import sde_lib
import sampling
import util.utils as utils
from models import ncsnpp
import models.utils as mutils
import torch
from torchvision.utils import make_grid
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import gc
import cog
import yaml


@dataclasses.dataclass
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
    name: str = "ncsnpp"
    normalization: str = "GroupNorm"
    nonlinearity: str = "swish"
    n_channels: str = 128
    ch_mult: str = "1,2,2,2"
    attn_resolutions: str = "16"
    resamp_with_conv: bool = True
    use_fir: bool = True
    fir_kernel: str = "1,3,3,1"
    skip_rescale: bool = True
    resblock_type: str = "biggan"
    progressive: Optional[str] = None
    progressive_input: str = "residual"
    progressive_combine: str = "sum"
    attention_type: str = "ddpm"
    init_scale: float = 0.0
    fourier_scale: int = 16
    conv_size: int = 3
    embedding_type: str = "fourier"
    mixed_score: bool = True
    n_resblocks: int = 8
    ema_rate: float = 0.9999
    numerical_eps: float = 1e-9
    sde: str = "cld"
    beta_type: str = "linear"
    beta0: float = 4.0
    beta1: float = 0.0
    m_inv: float = 4.0
    gamma: float = 0.04
    # Optimization
    optimizer: str = "Adam"
    learning_rate: float = 2e-4
    grad_clip: float = 1.0
    dropout: float = 0.1
    weight_decay: float = 0.0
    # Objective
    cld_objective: str = "hsm"
    loss_eps: float = 1e-5
    weighting: str = "reweightedv2"
    checkpoint: str = "checkpoints/cifar10_800000.pth"

    @classmethod
    def from_config(cls, config):
        return cls(**{k: str(v) for k, v in config.items()})


@dataclasses.dataclass
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
    with open(f'configs/{model}_cog.yaml', mode='r') as config_file:
        config = yaml.safe_load(config_file)
    model_config = ModelConfig(**config)
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