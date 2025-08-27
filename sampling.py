import jax
import jax.numpy as jnp


def make_sampler_fn(sde, denoiser_fn, shape, num_steps):
    def sampler_fn(
        rng,
        params,
        sampler='heun',
        s_churn=0.0,
        s_min=0.0,
        s_max=float("inf"),
        s_noise=1.0,
    ):
        step_indices = jnp.arange(num_steps)

        t_steps = (sde.sigma_max ** (1. / sde.rho) + (step_indices / (num_steps - 1))
                   * (sde.sigma_min ** (1. / sde.rho) - sde.sigma_max ** (1. / sde.rho))) ** sde.rho
        t_steps = jnp.concatenate([t_steps, jnp.zeros_like(t_steps[:1])])

        rng, sample_rng = jax.random.split(rng, 2)

        x = jax.random.normal(sample_rng, shape=shape) * t_steps[0]

        for i in range(num_steps):
            t_curr = t_steps[i]
            t_next = t_steps[i + 1]

            gamma = (jnp.minimum(s_churn / num_steps, jnp.sqrt(2.0) - 1.0)
                    * jnp.where((t_curr >= s_min) & (t_curr <= s_max), 1.0, 0.0))
            sigma_hat = t_curr * (1.0 + gamma)

            rng, sample_rng = jax.random.split(rng)
            eps = jax.random.normal(sample_rng, shape=x.shape) * s_noise

            x_hat = x + jnp.sqrt(jnp.maximum(sigma_hat**2 - t_curr**2, 0.0)) * eps

            x_denoised_hat = denoiser_fn(params, x_hat, sigma_hat)
            d = (x_hat - x_denoised_hat) / jnp.maximum(sigma_hat, 1e-12)

            x_euler = x_hat + (t_next - sigma_hat) * d

            needs_heun = jnp.asarray(t_next > 0.0, dtype=jnp.float32)

            x_denoised_next = denoiser_fn(params, x_euler, t_next)
            d_next = (x_euler - x_denoised_next) / jnp.maximum(t_next, 1e-12)

            x = needs_heun * (x_hat + (t_next - sigma_hat) * 0.5 * (d + d_next)) + (1.0 - needs_heun) * x_euler

        return x

    return jax.pmap(sampler_fn, axis_name='batch', donate_argnums=())
