import jax
import jax.numpy as jnp


class KarrasVESDE(object):

    def __init__(self, sigma_min=0.02, sigma_max=80, sigma_data=0.5, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def marginal_prob(self, x, t):
        mean = x
        std = t
        return mean, std

    def get_scalings(self, sigmas):
        c_in = 1 / jnp.sqrt(sigmas ** 2 + self.sigma_data ** 2)
        c_out = self.sigma_data * (sigmas - self.sigma_min) / jnp.sqrt(self.sigma_data ** 2 + sigmas ** 2)
        c_skip = self.sigma_data ** 2 / ((sigmas - self.sigma_min) ** 2 + self.sigma_data ** 2)
        return c_in, c_out, c_skip
