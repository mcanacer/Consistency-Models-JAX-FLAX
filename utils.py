import jax
import jax.numpy as jnp


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def get_denoiser_fn(apply_fn, sde):
    def denoiser_fn(params, x, t):
        c_in, c_out, c_skip = sde.get_scalings(t)
        rescaled_t = 1000 * 0.25 * jnp.log(t + 1e-44)
        rescaled_t = jnp.ones((x.shape[0],)) * rescaled_t
        in_x = batch_mul(c_in, x)
        pred = apply_fn(params, in_x, rescaled_t)

        denoiser = batch_mul(c_skip, x) + batch_mul(c_out, pred)

        return denoiser

    return denoiser_fn

