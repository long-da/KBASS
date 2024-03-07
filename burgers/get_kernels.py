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

    def get_derivative_cov(self, X, ls):
        num_X = X.shape[0]
        x_p1 = jnp.tile(X[:, 0].flatten(), (num_X, 1)).T
        X1_p1 = x_p1.flatten()
        X2_p1 = jnp.transpose(x_p1).flatten()

        x_p2 = jnp.tile(X[:, 1].flatten(), (num_X, 1)).T
        X1_p2 = x_p2.flatten()
        X2_p2 = jnp.transpose(x_p2).flatten()

        K_u1 = vmap(self.K_u.kappa, (0, 0, None))(X1_p1.flatten(), X2_p1.flatten(), ls[0]).reshape(num_X, num_X)
        K_u2 = vmap(self.K_u.kappa, (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(), ls[1]).reshape(num_X, num_X)
        K_dx1 = vmap(self.K_u.D_x1_kappa, (0, 0, None))(X1_p1.flatten(), X2_p1.flatten(), ls[0]).reshape(num_X, num_X)
        K_dx2 = vmap(self.K_u.D_x1_kappa, (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(), ls[1]).reshape(num_X, num_X)
        K_ddx1 = vmap(self.K_u.DD_x1_kappa, (0, 0, None))(X1_p1.flatten(), X2_p1.flatten(), ls[0]).reshape(num_X, num_X)
        K_ddx2 = vmap(self.K_u.DD_x1_kappa, (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(), ls[1]).reshape(num_X, num_X)
        K_dddx1 = vmap(self.K_u.DDD_x1_kappa, (0, 0, None))(X1_p1.flatten(), X2_p1.flatten(), ls[0]).reshape(num_X, num_X)
        K_dddx2 = vmap(self.K_u.DDD_x1_kappa, (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(), ls[1]).reshape(num_X, num_X)
        u = [K_u1, K_u2]
        u_dx1 = [K_dx1, K_u2]
        u_dx2 = [K_u1, K_dx2]
        u_ddx1 = [K_ddx1, K_u2]
        u_ddx2 = [K_u1, K_ddx2]
        u_dddx1 = [K_dddx1, K_u2]
        u_dddx2 = [K_u1, K_dddx2]

        return [u, u_dx1, u_dx2, u_ddx1, u_ddx2, u_dddx1, u_dddx2]

    def get_cov(self, X1, X2, ls):
        num_X1 = X1.shape[0]
        num_X2 = X2.shape[0]
        x1_p = jnp.tile(X1.flatten(), (num_X2, 1)).T
        x2_p = jnp.tile(X2.flatten(), (num_X1, 1)).T
        X1_p = x1_p.flatten()
        X2_p = jnp.transpose(x2_p).flatten()
        cov = vmap(self.K_u.kappa, (0, 0, None))(X1_p.flatten(), X2_p.flatten(), ls).reshape(num_X1, num_X2)
        return cov
