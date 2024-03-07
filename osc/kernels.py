import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from jax.config import config
import math

config.update("jax_enable_x64", True)


class RBF_kernel_u_1d(object):
    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, s1):
        return (jnp.exp(-1 / 2 * ((x1 - y1)**2 / s1**2))).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, s1):
        val = grad(self.kappa, 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 0), 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DDD_x1_kappa(self, x1, y1, s1):
        val = grad(grad(grad(self.kappa, 0), 0), 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DDDD_x1_kappa(self, x1, y1, s1):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 0), 0)(x1, y1, s1)
        return val

   