"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start,beta_end,T).to(torch.float32)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1.-betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    return a[t].reshape(-1,1,1,1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    bar_alpha_t = alphas_cumprod[t].reshape(-1,1,1,1) #(B,1,1,1)
    x_t = bar_alpha_t**0.5 * x0 + (1-bar_alpha_t)**0.5 * noise
    return x_t

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = linear_beta_schedule(T,beta_start,beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod =  (1-alphas_cumprod)**0.5
    return {
        'T': T,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return ((noise_pred-noise)**2).mean()

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    predicted_noise = model(x_t,t)
    loss = noise_prediction_loss(predicted_noise,noise)
    return loss

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half = dim//2
    base = 10000.
    i = torch.arange(half)[None,:]
    f = t[:,None] * base**(-i/half)
    sin = torch.sin(f)
    cos = torch.cos(f)
    return torch.cat((sin,cos),1)

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)
    return {
        'conv_in_w': (0.02*torch.randn(hidden,in_ch,3,3)).requires_grad_(True),
        'conv_in_b': torch.zeros(hidden).requires_grad_(True),
        'time_mlp_w': (0.02 * torch.randn(hidden, time_dim)).requires_grad_(True),
        'time_mlp_b': torch.zeros(hidden).requires_grad_(True),
        'conv_mid_w': (0.02 * torch.randn(hidden, hidden, 3, 3)).requires_grad_(True),
        'conv_mid_b': torch.zeros(hidden).requires_grad_(True),
        'conv_out_w': (0.02 * torch.randn(in_ch, hidden, 3, 3)).requires_grad_(True),
        'conv_out_b': torch.zeros(in_ch).requires_grad_(True),
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    h = F.conv2d(x, params["conv_in_w"], params['conv_in_b'],padding=1)
    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))
    temb = temb[...,None,None]
    h = h + temb
    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'],padding=1))
    return F.conv2d(h,params['conv_out_w'],params['conv_out_b'],padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    r = size//4
    c = torch.randint(r,size-r,(n,2))
    x = torch.arange(size)[None,None,:]
    y = torch.arange(size)[None,:,None]
    cx = c[:,0,None,None] #n,1,1
    cy = c[:,1,None,None]
    x_dist_sq = (x-cx)**2 #n,1,size
    y_dist_sq = (y-cy)**2 #n,size,1
    mask  = x_dist_sq+y_dist_sq <= r**2 #n,size,size
    return mask[:,None,:,:].float()

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    torch.manual_seed(seed)
    T = schedule['T']
    alphas_cumprod = schedule['alphas_cumprod']
    B = x0.shape[0]
    t = torch.randint(0,T,(B,))
    noise = torch.randn_like(x0)
    loss = diffusion_training_loss(lambda x,t: tiny_unet_forward(x,t,params), x0, t, noise, alphas_cumprod)
    loss.backward()
    for name,p in params.items():
        if p.grad is not None:
            params[name] = (p-lr*p.grad).detach().requires_grad_(True)
        else:
            params[name] = p.detach().clone().requires_grad_(True)
    return (params, float(loss))

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    # TODO: minibatch SGD training loop
    n = dataset.shape[0]
    history = []
    for i in range(num_steps):
        g = torch.Generator().manual_seed(seed+i)
        idx = torch.randint(0,n,(batch_size,),generator=g)
        mini_batch = dataset[idx,:]
        params, loss = ddpm_train_step(params,mini_batch,schedule,lr,seed+i)
        history.append(loss)
    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    alphas_bar = alphas_cumprod[t][:,None,None,None]
    x0 = 1/(alphas_bar)**0.5 * (x_t - (1-alphas_bar)**0.5 * eps)
    return x0

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # TODO: return (posterior_mean, variance, x0_hat)
    x0_hat = torch.clip(predict_x0_from_eps(x_t,t,eps,schedule['alphas_cumprod']),0.,1.)
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    betas = schedule['betas']
    posterior_mean = 1/(alphas[t][:,None,None,None])**0.5 * (x_t - betas[t][:,None,None,None]/(1-alphas_cumprod[t][:,None,None,None])**0.5 * eps)
    variance = betas[t][:,None,None,None]
    return posterior_mean, variance, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # TODO: one reverse step x_t -> x_{t-1}
    eps = tiny_unet_forward(x_t,t,params)
    mean,var,_ = ddpm_p_mean_variance(x_t,t,eps,schedule)
    if noise is None:
        noise = torch.randn_like(x_t)
    noise = (t!=0)[:,None,None,None].float() * noise
    x_prev = mean + var**0.5 * noise
    return x_prev

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # TODO: ancestral sampling from pure noise to x0
    torch.manual_seed(seed)
    x = torch.randn(shape)
    for t in range(schedule['T']-1,0,):
        t_batch = torch.tensor([t]*shape[0])
        x = ddpm_p_sample(x,t_batch,params,schedule)
    return x

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

