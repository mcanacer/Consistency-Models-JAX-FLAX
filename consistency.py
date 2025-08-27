import sys
import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from unet import UNet
from lpips import LPIPS
import sde_lib
from utils import batch_mul
from dataset import HuggingFace
from datasets import load_dataset

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def make_ema_and_scale_fn(target_ema_mode, start_ema, scale_mode, start_scales, end_scales, total_steps):
    if target_ema_mode == "fixed" and scale_mode == "fixed":
        def step_scheduler(step):
            return start_ema, start_scales
    elif target_ema_mode == 'adaptive' and scale_mode == 'progressive':
        def step_scheduler(step):
            scales = jnp.ceil(
                jnp.sqrt(
                    (step / total_steps)
                    * ((end_scales + 1) ** 2 - start_scales ** 2)
                    + start_scales ** 2
                )
                - 1
            ).astype(jnp.int32)
            scales = jnp.maximum(scales, 1)
            c = -jnp.log(start_ema) * start_scales
            target_ema = jnp.exp(-c / scales)
            scales = scales + 1
            return target_ema, scales
    else:
        raise NotImplementedError
    return step_scheduler


def get_denoiser_fn(apply_fn, sde):
    def denoiser_fn(params, x, t):
        c_in, c_out, c_skip = sde.get_scalings(t)
        rescaled_t = 1000 * 0.25 * jnp.log(t + 1e-44)
        in_x = batch_mul(c_in, x)
        pred = apply_fn(params, in_x, rescaled_t)

        denoiser = batch_mul(c_skip, x) + batch_mul(c_out, pred)

        return denoiser

    return denoiser_fn


def make_update_fn(*, optimizer, denoiser_fn, lpips_apply_fn, sde, loss_norm, ema_decay):
    def update_fn(params, opt_state, x, rng, teacher_params, lpips_params, target_params, num_scales, target_ema_decay, ema_params):
        def heun_solver(params, samples, t, next_t, x0=None):
            x = samples
            if params is None:
                denoiser = x0
            else:
                denoiser = denoiser_fn(params, x, t)

            d = batch_mul((x - denoiser), 1 / t)
            samples = x + batch_mul(d, (next_t - t))
            if params is None:
                denoiser = x0
            else:
                denoiser = denoiser_fn(params, samples, next_t)

            next_d = batch_mul((samples - denoiser), 1 / next_t)
            samples = x + batch_mul((d + next_d), (next_t - t) / 2)

            return samples

        def euler_solver(params, samples, t, next_t, x0=None):
            x = samples
            if params is None:
                denoiser = x0
            else:
                denoiser = denoiser_fn(params, x, t)

            d = batch_mul((x - denoiser), 1 / t)
            samples = x + batch_mul(d, (next_t - t))

            return samples

        def loss_fn(params):
            time_rng, sample_rng = jax.random.split(rng, 2)

            indices = jax.random.randint(time_rng, shape=(x.shape[0],), minval=0, maxval=num_scales - 1)

            t = sde.sigma_max ** (1 / sde.rho) + indices / (num_scales - 1) * (
                    sde.sigma_min ** (1 / sde.rho) - sde.sigma_max ** (1 / sde.rho)
            )

            t = t ** sde.rho

            t2 = sde.sigma_max ** (1 / sde.rho) + (indices + 1) / (num_scales - 1) * (
                    sde.sigma_min ** (1 / sde.rho) - sde.sigma_max ** (1 / sde.rho)
            )
            t2 = t2 ** sde.rho

            z = jax.random.normal(sample_rng, shape=x.shape)
            x_t = x + batch_mul(t, z)

            distiller = denoiser_fn(params, x_t, t)

            if teacher_params is None:
                x_t2 = euler_solver(teacher_params, x_t, t, t2, x)
            else:
                x_t2 = heun_solver(teacher_params, x_t, t, t2, x)

            distiller_target = denoiser_fn(target_params, x_t2, t2)

            distiller_target = jax.lax.stop_gradient(distiller_target)

            if loss_norm == 'l1':
                loss = jnp.mean(jnp.abs(distiller - distiller_target))
            elif loss_norm == 'l2':
                loss = jnp.mean((distiller - distiller_target) ** 2)
            elif loss_norm == 'lpips':
                distiller = jax.image.resize(
                    distiller, (distiller.shape[0], 224, 224, 3), method="bilinear"
                )
                distiller_target = jax.image.resize(
                    distiller_target, (distiller_target.shape[0], 224, 224, 3), method="bilinear"
                )
                loss = jnp.squeeze(lpips_apply_fn(lpips_params, distiller, distiller_target).sum())
            else:
                raise ValueError(f'Unkown loss norm: {loss_norm}')

            return loss

        loss, grad = jax.value_and_grad(loss_fn)(params)

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_target_params = ema_update(target_params, new_params, decay=target_ema_decay)
        new_ema_params = ema_update(ema_params, new_params, decay=ema_decay)

        return new_params, opt_state, new_target_params, new_ema_params, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    model_config = config['model']
    teacher_config = config['teacher_model']
    ema_and_scale_config = config['ema_and_scale']
    sde_config = config['SDE']
    dataset_params = config['dataset_params']
    wandb_config = config['wandb']

    seed = model_config['seed']

    transform = transforms.Compose([
        transforms.Resize((dataset_params['img_size'], dataset_params['img_size'])),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.RandomHorizontalFlip(0.5),
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    train_dataset = HuggingFace(
        dataset=load_dataset("pcuenq/lsun-bedrooms", split='train'),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_params['batch_size'],
        shuffle=True,
        num_workers=dataset_params['num_workers'],
        drop_last=True,
    )

    model = UNet(**model_config['params'])
    lpips = LPIPS()
    sde = sde_lib.KarrasVESDE(**sde_config['params'])

    optimizer = optax.chain(
        optax.radam(
            learning_rate=float(model_config['learning_rate']),
            b1=model_config['b1']
        )
    )

    epochs = model_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = model_config['checkpoint_path']

    key = jax.random.PRNGKey(seed)
    key, init_key, sub_key = jax.random.split(key, 3)
    lpips_params = lpips.init(init_key, jnp.ones((2, 224, 224, 3)), jnp.ones((2, 224, 224, 3)))

    if teacher_config['checkpoint_path']:
        teacher_params = load_checkpoint(teacher_config['checkpoint'], None)['ema_params']

        model_params = teacher_params
        target_params = teacher_params
    else:
        teacher_params = None

        model_params = model.init(sub_key, jnp.ones((2, 128, 128, 3)), jnp.ones((2,), dtype=jnp.int32))
        target_params = model_params

    opt_state = optimizer.init(model_params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    ema_decay = model_config['ema_decay']

    model_ema_params = model_params
    model_ema_params_repl = replicate(model_ema_params)
    teacher_params_repl = replicate(teacher_params) if teacher_params is not None else None

    denoiser_fn = get_denoiser_fn(apply_fn=model.apply, sde=sde)

    update_fn = make_update_fn(
        optimizer=optimizer,
        denoiser_fn=denoiser_fn,
        lpips_apply_fn=lpips.apply,
        sde=sde,
        loss_norm='lpips',
        ema_decay=ema_decay,
    )

    ema_and_scale_fn = make_ema_and_scale_fn(**ema_and_scale_config)

    model_params_repl = replicate(model_params)
    lpips_params_repl = replicate(lpips_params)
    target_params_repl = replicate(target_params)
    opt_state_repl = replicate(opt_state)

    global_step = 0

    del model_params
    del opt_state

    num_devices = jax.local_device_count()

    state_template = {
        "global_step": global_step,
        "params": unreplicate(model_params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "ema_params": unreplicate(model_ema_params_repl),
        "target_params": unreplicate(target_params_repl),
        "epoch": 0,
        "rng": key,
    }

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    if loaded_state is not None:
        print("Resuming from checkpoint...")
        model_params_repl = replicate(loaded_state["params"])
        opt_state_repl = replicate(loaded_state["opt_state"])
        model_ema_params_repl = replicate(loaded_state["ema_params"])
        target_params_repl = replicate(loaded_state['target_params'])
        key = loaded_state["rng"]
        start_epoch = loaded_state["epoch"] + 1
    else:
        start_epoch = 0

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(start_epoch, epochs):
        for images in train_loader:
            key, step_rng = jax.random.split(key, 2)

            target_ema_decay, num_scales = ema_and_scale_fn(global_step)

            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            rng_shard = jax.random.split(step_rng, num_devices)

            (
                model_params_repl,
                opt_state_repl,
                target_params_repl,
                model_ema_params_repl,
                loss,
            ) = update_fn(
                model_params_repl,
                opt_state_repl,
                images,
                rng_shard,
                teacher_params_repl,
                lpips_params_repl,
                target_params_repl,
                replicate(num_scales),
                replicate(target_ema_decay),
                model_ema_params_repl,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

            global_step += 1

        checkpoint_state = {
            "params": unreplicate(model_params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "ema_params": unreplicate(model_ema_params_repl),
            "target_params": unreplicate(target_params_repl),
            "epoch": epoch,
            "rng": key,
        }
        save_checkpoint(checkpoint_path, checkpoint_state)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
