import jax.numpy as jnp
from jax import vmap
import numpy as np
from kernels import *


class Kernel(object):

    def __init__(self, jitter, K_u):
        self.jitter = jitter
        self.K_u = K_u

    def get_kernel_matrix(self, X, ls):
        num_X = X.shape[0]
        x_p = jnp.tile(X.flatten(), (num_X, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        N = int((X1_p.shape[0])**0.5)
        K_u_u = vmap(self.K_u.kappa, (0, 0, None))(X1_p.flatten(), X2_p.flatten(), ls).reshape(N, N)
        K_u_u = K_u_u + self.jitter * jnp.eye(N)
        return K_u_u

    def get_cov(self, X1, X2, ls):
        num_X1 = X1.shape[0]
        num_X2 = X2.shape[0]
        x1_p = jnp.tile(X1.flatten(), (num_X2, 1)).T
        x2_p = jnp.tile(X2.flatten(), (num_X1, 1)).T
        X1_p = x1_p.flatten()
        X2_p = jnp.transpose(x2_p).flatten()
        cov = vmap(self.K_u.kappa, (0, 0, None))(X1_p.flatten(), X2_p.flatten(), ls).reshape(num_X1, num_X2)
        return cov

    def get_cov(self, X1, X2, ls):
        num_X1 = X1.shape[0]
        num_X2 = X2.shape[0]
        x1_p = jnp.tile(X1.flatten(), (num_X2, 1)).T
        x2_p = jnp.tile(X2.flatten(), (num_X1, 1)).T
        X1_p = x1_p.flatten()
        X2_p = jnp.transpose(x2_p).flatten()
        cov = vmap(self.K_u.kappa, (0, 0, None))(X1_p.flatten(), X2_p.flatten(), ls).reshape(num_X1, num_X2)
        return cov

    def get_derivative_cov_1d(self, X, ls):
        num_X = X.shape[0]
        x_p1 = jnp.tile(X[:, 0].flatten(), (num_X, 1)).T
        X1_p1 = x_p1.flatten()
        X2_p1 = jnp.transpose(x_p1).flatten()
        K_u1 = vmap(self.K_u.kappa, (0, 0, None))(X1_p1.flatten(), X2_p1.flatten(), ls).reshape(num_X, num_X)
        K_dx1 = vmap(self.K_u.D_x1_kappa, (0, 0, None))(X1_p1.flatten(), X2_p1.flatten(), ls).reshape(num_X, num_X)
        u = [K_u1]
        u_dx1 = [K_dx1]
        return [u, u_dx1]
