import sys
import yaml
import os

import torch
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization
from torchvision.utils import save_image
from unet import UNet
from sde_lib import KarrasVESDE
from utils import get_denoiser_fn, batch_mul


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def generate_samples(
    rng,
    unet,
    unet_params,
    sde,
    num_steps,
    shape,
    s_churn=0.0,
    s_min=0.0,
    s_max=float("inf"),
    s_noise=1.0,
):
    import functools
    devices = jax.local_devices()
    num_devices = len(devices)

    per_device_batch = shape[0] // num_devices

    def shard(x):
        x = np.asarray(x)
        return x.reshape(num_devices, per_device_batch, *x.shape[1:])

    def unshard(x):
        x = np.asarray(x)
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

    replicate = lambda tree: jax.device_put_replicated(tree, devices)

    denoiser_fn = get_denoiser_fn(apply_fn=unet.apply, sde=sde)

    @functools.partial(jax.pmap, axis_name='devices')
    def p_predict_fn(params_repl, x_shard, sigma_shard):
        if sigma_shard.ndim == 0:
            sigma_shard = jnp.full((x_shard.shape[0],), sigma_shard)
        return denoiser_fn(params_repl, x_shard, sigma_shard)

    ramp = jnp.linspace(0, 1, num_steps)
    min_inv_rho = sde.sigma_min ** (1 / sde.rho)
    max_inv_rho = sde.sigma_max ** (1 / sde.rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** sde.rho
    sigmas = jnp.concatenate([sigmas, jnp.zeros_like(sigmas[:1])], axis=0)

    rng, sample_rng = jax.random.split(rng)
    x = jax.random.normal(sample_rng, shape=shape) * sde.sigma_max

    params_repl = replicate(unet_params)

    for i in range(num_steps):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = jnp.minimum(s_churn / num_steps, jnp.sqrt(2.0) - 1.0)
        gamma = gamma * jnp.where((sigma_curr >= s_min) & (sigma_curr <= s_max), 1.0, 0.0)
        sigma_hat = sigma_curr * (1.0 + gamma)

        rng, eps_rng = jax.random.split(rng)
        eps = jax.random.normal(eps_rng, shape=x.shape) * s_noise
        delta = jnp.maximum(sigma_hat**2 - sigma_curr**2, 0.0)
        x_hat = x + jnp.sqrt(delta) * eps

        shard_x = shard(x_hat)
        shard_sigma_hat = shard(jnp.full((shape[0],), sigma_hat))
        x_denoised_hat = p_predict_fn(params_repl, shard_x, shard_sigma_hat)
        x_denoised_hat = unshard(x_denoised_hat)

        denom_hat = jnp.maximum(sigma_hat, 1e-12)
        d = (x_hat - x_denoised_hat) / denom_hat
        x_euler = x_hat + (sigma_next - sigma_hat) * d

        if sigma_next != 0:

            shard_x2 = shard(x_euler)
            shard_sigma_next = shard(jnp.full((shape[0],), sigma_next))
            x_denoised_next = p_predict_fn(params_repl, shard_x2, shard_sigma_next)
            x_denoised_next = unshard(x_denoised_next)

            denom_next = jnp.maximum(sigma_next, 1e-12)
            d_next = (x_euler - x_denoised_next) / denom_next
            x = (x_hat + (sigma_next - sigma_hat) * 0.5 * (d + d_next))

    x = (x + 1.0) / 2.0
    return jnp.clip(x, 0.0, 1.0)


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    unet_config = config['model']
    sde_config = config['SDE']

    seed = unet_config['seed']
    key = jax.random.PRNGKey(seed)

    unet = UNet(**unet_config['params'])
    sde = KarrasVESDE(**sde_config['params'])

    checkpoint_path = unet_config['checkpoint_path']
    unet_params = load_checkpoint(checkpoint_path, None)['ema_params']

    x_gen = generate_samples(key, unet, unet_params, sde, 40, (64, 128, 128, 3))

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/consistency-models/gen_images/generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
