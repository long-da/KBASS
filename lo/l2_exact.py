from jax.lib import xla_bridge
from numpy import random
import numpy as np
from kernels import *
from get_kernels import *
import random
import optax
import jax
import jax.numpy as jnp
import tensorly as tl
from scipy.stats import norm as normal
from scipy.special import *
from scipy.io import loadmat
import os
from scipy.io import savemat

np.random.seed(0)
random.seed(0)
print("Jax on", xla_bridge.get_backend().platform)
tl.set_backend('jax')


class EQD(object):

    def __init__(self, sensors, Y_train, X_test, Y_test):
        self.sensors = sensors
        self.Y1 = Y_train[:, 0].reshape(-1, 1)
        self.Y2 = Y_train[:, 1].reshape(-1, 1)
        self.Y3 = Y_train[:, 2].reshape(-1, 1)
        self.Y4 = Y_train[:, 3].reshape(-1, 1)
        self.Y5 = Y_train[:, 4].reshape(-1, 1)
        self.Y6 = Y_train[:, 5].reshape(-1, 1)
        self.jitter = 1e-6
        self.num = 180
        self.d = 1
        self.X_test = X_test
        self.Y_test = Y_test
        self.lb = X_test.min(axis=0)
        self.ub = X_test.max(axis=0)
        self.X_col = np.array([np.linspace(self.lb[i], self.ub[i], self.num) for i in range(self.lb.shape[0])])
        self.N = 12
        self.N_col = (self.num)**self.d
        self.num_operators = 72
        self.th = 0.2
        self.bound = 1e-5
        self.init()

    def init(self):
        self.tau1 = 0.01
        self.rho1 = logit(0.5)
        self.tol1 = 1e-5
        self.damping1 = 0.9
        self.s01 = 1.0
        self.INF1 = 1000000.0
        self.rho_p_full1 = np.zeros(self.num_operators)
        self.mu_p_full1 = np.zeros(self.num_operators)
        self.v_p_full1 = self.INF1 * np.ones(self.num_operators)
        self.r_full1 = self.rho_p_full1 + self.rho1

        self.tau2 = 0.01
        self.rho2 = logit(0.5)
        self.tol2 = 1e-5
        self.damping2 = 0.9
        self.s02 = 1.0
        self.INF2 = 1000000.0
        self.rho_p_full2 = np.zeros(self.num_operators)
        self.mu_p_full2 = np.zeros(self.num_operators)
        self.v_p_full2 = self.INF2 * np.ones(self.num_operators)
        self.r_full2 = self.rho_p_full2 + self.rho2

        self.tau3 = 0.01
        self.rho3 = logit(0.5)
        self.tol3 = 1e-5
        self.damping3 = 0.9
        self.s03 = 1.0
        self.INF3 = 1000000.0
        self.rho_p_full3 = np.zeros(self.num_operators)
        self.mu_p_full3 = np.zeros(self.num_operators)
        self.v_p_full3 = self.INF3 * np.ones(self.num_operators)
        self.r_full3 = self.rho_p_full3 + self.rho3

        self.tau4 = 0.01
        self.rho4 = logit(0.5)
        self.tol4 = 1e-5
        self.damping4 = 0.9
        self.s04 = 1.0
        self.INF4 = 1000000.0
        self.rho_p_full4 = np.zeros(self.num_operators)
        self.mu_p_full4 = np.zeros(self.num_operators)
        self.v_p_full4 = self.INF4 * np.ones(self.num_operators)
        self.r_full4 = self.rho_p_full4 + self.rho4

        self.tau5 = 0.01
        self.rho5 = logit(0.5)
        self.tol5 = 1e-5
        self.damping5 = 0.9
        self.s05 = 1.0
        self.INF5 = 1000000.0
        self.rho_p_full5 = np.zeros(self.num_operators)
        self.mu_p_full5 = np.zeros(self.num_operators)
        self.v_p_full5 = self.INF5 * np.ones(self.num_operators)
        self.r_full5 = self.rho_p_full5 + self.rho5

        self.tau6 = 0.01
        self.rho6 = logit(0.5)
        self.tol6 = 1e-5
        self.damping6 = 0.9
        self.s06 = 1.0
        self.INF6 = 1000000.0
        self.rho_p_full6 = np.zeros(self.num_operators)
        self.mu_p_full6 = np.zeros(self.num_operators)
        self.v_p_full6 = self.INF6 * np.ones(self.num_operators)
        self.r_full6 = self.rho_p_full6 + self.rho6

    @partial(jit, static_argnums=(0, ))
    def get_lib(self, params):
        a = 45
        b = 120
        mu = params['mu'].sum(axis=-1)
        mu2 = params['mu2'].sum(axis=-1)
        mu3 = params['mu3'].sum(axis=-1)
        mu4 = params['mu4'].sum(axis=-1)
        mu5 = params['mu5'].sum(axis=-1)
        mu6 = params['mu6'].sum(axis=-1)
        u_sample = mu
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        self.derivative_cov_list = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j, :]) for j in range(len(self.derivative_cov_list[0]))] for i in range(len(self.derivative_cov_list))]
        deri_sample = [tl.tenalg.multi_mode_dot(u_sample, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        ls2 = jnp.exp(params['log_ls2'])
        self.Kernel2 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list2 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list2 = jnp.linalg.inv(self.K_list2)
        self.derivative_cov_list2 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls2)
        self.deri_cov_times_K_inv_list2 = [[jnp.matmul(self.derivative_cov_list2[i][j], self.K_inv_list2[j, :]) for j in range(len(self.derivative_cov_list2[0]))] for i in range(len(self.derivative_cov_list2))]
        u_sample2 = mu2
        deri_sample2 = [tl.tenalg.multi_mode_dot(u_sample2, self.deri_cov_times_K_inv_list2[i]) for i in range(len(self.deri_cov_times_K_inv_list2))]
        u_dx1_2 = deri_sample2[1].reshape(1, -1)
        u_2 = deri_sample2[0].reshape(1, -1)
        ls3 = jnp.exp(params['log_ls3'])
        self.Kernel3 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list3 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls3[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list3 = jnp.linalg.inv(self.K_list3)
        self.derivative_cov_list3 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls3)
        self.deri_cov_times_K_inv_list3 = [[jnp.matmul(self.derivative_cov_list3[i][j], self.K_inv_list3[j, :]) for j in range(len(self.derivative_cov_list3[0]))] for i in range(len(self.derivative_cov_list3))]
        deri_sample3 = [tl.tenalg.multi_mode_dot(mu3, self.deri_cov_times_K_inv_list3[i]) for i in range(len(self.deri_cov_times_K_inv_list3))]
        u_dx1_3 = deri_sample3[1].reshape(1, -1)
        u_3 = deri_sample3[0].reshape(1, -1)
        ls4 = jnp.exp(params['log_ls4'])
        self.Kernel4 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list4 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls4[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list4 = jnp.linalg.inv(self.K_list4)
        self.derivative_cov_list4 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls4)
        self.deri_cov_times_K_inv_list4 = [[jnp.matmul(self.derivative_cov_list4[i][j], self.K_inv_list4[j, :]) for j in range(len(self.derivative_cov_list4[0]))] for i in range(len(self.derivative_cov_list4))]
        deri_sample4 = [tl.tenalg.multi_mode_dot(mu4, self.deri_cov_times_K_inv_list4[i]) for i in range(len(self.deri_cov_times_K_inv_list4))]
        u_dx1_4 = deri_sample4[1].reshape(1, -1)
        u_4 = deri_sample4[0].reshape(1, -1)
        ls5 = jnp.exp(params['log_ls5'])
        self.Kernel5 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list5 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls5[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list5 = jnp.linalg.inv(self.K_list5)
        self.derivative_cov_list5 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls5)
        self.deri_cov_times_K_inv_list5 = [[jnp.matmul(self.derivative_cov_list5[i][j], self.K_inv_list5[j, :]) for j in range(len(self.derivative_cov_list5[0]))] for i in range(len(self.derivative_cov_list5))]
        deri_sample5 = [tl.tenalg.multi_mode_dot(mu5, self.deri_cov_times_K_inv_list5[i]) for i in range(len(self.deri_cov_times_K_inv_list5))]
        u_dx1_5 = deri_sample5[1].reshape(1, -1)
        u_5 = deri_sample5[0].reshape(1, -1)
        ls6 = jnp.exp(params['log_ls6'])
        self.Kernel6 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list6 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls6[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list6 = jnp.linalg.inv(self.K_list6)
        self.derivative_cov_list6 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls6)
        self.deri_cov_times_K_inv_list6 = [[jnp.matmul(self.derivative_cov_list6[i][j], self.K_inv_list6[j, :]) for j in range(len(self.derivative_cov_list6[0]))] for i in range(len(self.derivative_cov_list6))]
        deri_sample6 = [tl.tenalg.multi_mode_dot(mu6, self.deri_cov_times_K_inv_list6[i]) for i in range(len(self.deri_cov_times_K_inv_list6))]
        u_dx1_6 = deri_sample6[1].reshape(1, -1)
        u_6 = deri_sample6[0].reshape(1, -1)
        ones = jnp.ones((1, self.num)) * 8
        library1 = jnp.concatenate((
            u_2 * u_6,
            u_5 * u_6,
            u,
            ones,
            u_2,
            u_3,
            u_4,
            u_5,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library2 = jnp.concatenate((
            u * u_3,
            u * u_6,
            u_2,
            ones,
            u,
            u_3,
            u_4,
            u_5,
            u_6,
            u * u_2,
            u * u_4,
            u * u_5,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library3 = jnp.concatenate((
            u_2 * u_4,
            u * u_2,
            u_3,
            ones,
            u,
            u_2,
            u_4,
            u_5,
            u_6,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library4 = jnp.concatenate((
            u_3 * u_5,
            u_2 * u_3,
            u_4,
            ones,
            u,
            u_2,
            u_3,
            u_5,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library5 = jnp.concatenate((
            u_4 * u_6,
            u_3 * u_4,
            u_5,
            ones,
            u,
            u_2,
            u_3,
            u_4,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library6 = jnp.concatenate((
            u * u_5,
            u_4 * u_5,
            u_6,
            ones,
            u,
            u_2,
            u_3,
            u_4,
            u_5,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        X1 = jnp.array(library1.T)
        y1 = jnp.array(u_dx1.reshape(-1))
        X2 = jnp.array(library2.T)
        y2 = jnp.array(u_dx1_2.reshape(-1))
        X3 = jnp.array(library3.T)
        y3 = jnp.array(u_dx1_3.reshape(-1))
        X4 = jnp.array(library4.T)
        y4 = jnp.array(u_dx1_4.reshape(-1))
        X5 = jnp.array(library5.T)
        y5 = jnp.array(u_dx1_5.reshape(-1))
        X6 = jnp.array(library6.T)
        y6 = jnp.array(u_dx1_6.reshape(-1))
        return X1[a:b, :], X2[a:b, :], X3[a:b, :], X4[a:b, :], X5[a:b, :], X6[a:b, :], y1[a:b], y2[a:b], y3[a:b], y4[a:b], y5[a:b], y6[a:b]

    @partial(jit, static_argnums=(0, ))
    def loss(self, params, co, w1, w2, w3, w4, w5, w6, tau, v):
        a = 45
        b = 120
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.cho_inv_list = jnp.linalg.inv(self.cho_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        mu = params['mu'].sum(axis=-1)
        self.cov_f = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j, :]) for j in range(len(self.derivative_cov_list[0]))] for i in range(len(self.derivative_cov_list))]
        self.cov_f_inv_K = [jnp.linalg.solve(self.K_list[i], self.cov_f[i].T).T for i in range(self.X_col.shape[0])]
        f_sample = tl.tenalg.multi_mode_dot(mu, self.cov_f_inv_K).reshape(-1)
        deri_sample = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        u_K_inv_u = ((tl.tenalg.multi_mode_dot(mu, self.cho_inv_list))**2).sum()
        ls2 = jnp.exp(params['log_ls2'])
        self.Kernel2 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list2 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        self.cho_list2 = jnp.linalg.cholesky(self.K_list2)
        self.cho_inv_list2 = jnp.linalg.inv(self.cho_list2)
        self.K_inv_list2 = jnp.linalg.inv(self.K_list2)
        mu2 = params['mu2'].sum(axis=-1)
        self.cov_f2 = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list2 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls2)
        self.deri_cov_times_K_inv_list2 = [[jnp.matmul(self.derivative_cov_list2[i][j], self.K_inv_list2[j, :]) for j in range(len(self.derivative_cov_list2[0]))] for i in range(len(self.derivative_cov_list2))]
        self.cov_f_inv_K2 = [jnp.linalg.solve(self.K_list2[i], self.cov_f2[i].T).T for i in range(self.X_col.shape[0])]
        f_sample2 = tl.tenalg.multi_mode_dot(mu2, self.cov_f_inv_K2).reshape(-1)
        deri_sample2 = [tl.tenalg.multi_mode_dot(mu2, self.deri_cov_times_K_inv_list2[i]) for i in range(len(self.deri_cov_times_K_inv_list2))]
        u_dx1_2 = deri_sample2[1].reshape(1, -1)
        u_2 = deri_sample2[0].reshape(1, -1)
        u_K_inv_u2 = ((tl.tenalg.multi_mode_dot(mu2, self.cho_inv_list2))**2).sum()
        u = u.reshape(1, -1)
        u_2 = u_2.reshape(1, -1)
        ls3 = jnp.exp(params['log_ls3'])
        self.Kernel3 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list3 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls3[i]) for i in range(self.X_col.shape[0])])
        self.cho_list3 = jnp.linalg.cholesky(self.K_list3)
        self.cho_inv_list3 = jnp.linalg.inv(self.cho_list3)
        self.K_inv_list3 = jnp.linalg.inv(self.K_list3)
        mu3 = params['mu3'].sum(axis=-1)
        self.cov_f3 = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls3[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list3 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls3)
        self.deri_cov_times_K_inv_list3 = [[jnp.matmul(self.derivative_cov_list3[i][j], self.K_inv_list3[j, :]) for j in range(len(self.derivative_cov_list3[0]))] for i in range(len(self.derivative_cov_list3))]
        self.cov_f_inv_K3 = [jnp.linalg.solve(self.K_list3[i], self.cov_f3[i].T).T for i in range(self.X_col.shape[0])]
        f_sample3 = tl.tenalg.multi_mode_dot(mu3, self.cov_f_inv_K3).reshape(-1)
        deri_sample3 = [tl.tenalg.multi_mode_dot(mu3, self.deri_cov_times_K_inv_list3[i]) for i in range(len(self.deri_cov_times_K_inv_list3))]
        u_dx1_3 = deri_sample3[1].reshape(1, -1)
        u_3 = deri_sample3[0].reshape(1, -1)
        u_K_inv_u3 = ((tl.tenalg.multi_mode_dot(mu3, self.cho_inv_list3))**2).sum()
        ls4 = jnp.exp(params['log_ls4'])
        self.Kernel4 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list4 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls4[i]) for i in range(self.X_col.shape[0])])
        self.cho_list4 = jnp.linalg.cholesky(self.K_list4)
        self.cho_inv_list4 = jnp.linalg.inv(self.cho_list4)
        self.K_inv_list4 = jnp.linalg.inv(self.K_list4)
        mu4 = params['mu4'].sum(axis=-1)
        self.cov_f4 = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls4[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list4 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls4)
        self.deri_cov_times_K_inv_list4 = [[jnp.matmul(self.derivative_cov_list4[i][j], self.K_inv_list4[j, :]) for j in range(len(self.derivative_cov_list4[0]))] for i in range(len(self.derivative_cov_list4))]
        self.cov_f_inv_K4 = [jnp.linalg.solve(self.K_list4[i], self.cov_f4[i].T).T for i in range(self.X_col.shape[0])]
        f_sample4 = tl.tenalg.multi_mode_dot(mu4, self.cov_f_inv_K4).reshape(-1)
        deri_sample4 = [tl.tenalg.multi_mode_dot(mu4, self.deri_cov_times_K_inv_list4[i]) for i in range(len(self.deri_cov_times_K_inv_list4))]
        u_dx1_4 = deri_sample4[1].reshape(1, -1)
        u_4 = deri_sample4[0].reshape(1, -1)
        u_K_inv_u4 = ((tl.tenalg.multi_mode_dot(mu4, self.cho_inv_list4))**2).sum()
        ls5 = jnp.exp(params['log_ls5'])
        self.Kernel5 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list5 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls5[i]) for i in range(self.X_col.shape[0])])
        self.cho_list5 = jnp.linalg.cholesky(self.K_list5)
        self.cho_inv_list5 = jnp.linalg.inv(self.cho_list5)
        self.K_inv_list5 = jnp.linalg.inv(self.K_list5)
        mu5 = params['mu5'].sum(axis=-1)
        self.cov_f5 = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls5[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list5 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls5)
        self.deri_cov_times_K_inv_list5 = [[jnp.matmul(self.derivative_cov_list5[i][j], self.K_inv_list5[j, :]) for j in range(len(self.derivative_cov_list5[0]))] for i in range(len(self.derivative_cov_list5))]
        self.cov_f_inv_K5 = [jnp.linalg.solve(self.K_list5[i], self.cov_f5[i].T).T for i in range(self.X_col.shape[0])]
        f_sample5 = tl.tenalg.multi_mode_dot(mu5, self.cov_f_inv_K5).reshape(-1)
        deri_sample5 = [tl.tenalg.multi_mode_dot(mu5, self.deri_cov_times_K_inv_list5[i]) for i in range(len(self.deri_cov_times_K_inv_list5))]
        u_dx1_5 = deri_sample5[1].reshape(1, -1)
        u_5 = deri_sample5[0].reshape(1, -1)
        u_K_inv_u5 = ((tl.tenalg.multi_mode_dot(mu5, self.cho_inv_list5))**2).sum()
        ls6 = jnp.exp(params['log_ls6'])
        self.Kernel6 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list6 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls6[i]) for i in range(self.X_col.shape[0])])
        self.cho_list6 = jnp.linalg.cholesky(self.K_list6)
        self.cho_inv_list6 = jnp.linalg.inv(self.cho_list6)
        self.K_inv_list6 = jnp.linalg.inv(self.K_list6)
        mu6 = params['mu6'].sum(axis=-1)
        self.cov_f6 = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls6[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list6 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls6)
        self.deri_cov_times_K_inv_list6 = [[jnp.matmul(self.derivative_cov_list6[i][j], self.K_inv_list6[j, :]) for j in range(len(self.derivative_cov_list6[0]))] for i in range(len(self.derivative_cov_list6))]
        self.cov_f_inv_K6 = [jnp.linalg.solve(self.K_list6[i], self.cov_f6[i].T).T for i in range(self.X_col.shape[0])]
        f_sample6 = tl.tenalg.multi_mode_dot(mu6, self.cov_f_inv_K6).reshape(-1)
        deri_sample6 = [tl.tenalg.multi_mode_dot(mu6, self.deri_cov_times_K_inv_list6[i]) for i in range(len(self.deri_cov_times_K_inv_list6))]
        u_dx1_6 = deri_sample6[1].reshape(1, -1)
        u_6 = deri_sample6[0].reshape(1, -1)
        u_K_inv_u6 = ((tl.tenalg.multi_mode_dot(mu6, self.cho_inv_list6))**2).sum()
        ones = jnp.ones((1, self.num)) * 8
        library1 = jnp.concatenate((
            u_2 * u_6,
            u_5 * u_6,
            u,
            ones,
            u_2,
            u_3,
            u_4,
            u_5,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library2 = jnp.concatenate((
            u * u_3,
            u * u_6,
            u_2,
            ones,
            u,
            u_3,
            u_4,
            u_5,
            u_6,
            u * u_2,
            u * u_4,
            u * u_5,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library3 = jnp.concatenate((
            u_2 * u_4,
            u * u_2,
            u_3,
            ones,
            u,
            u_2,
            u_4,
            u_5,
            u_6,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library4 = jnp.concatenate((
            u_3 * u_5,
            u_2 * u_3,
            u_4,
            ones,
            u,
            u_2,
            u_3,
            u_5,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library5 = jnp.concatenate((
            u_4 * u_6,
            u_3 * u_4,
            u_5,
            ones,
            u,
            u_2,
            u_3,
            u_4,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library6 = jnp.concatenate((
            u * u_5,
            u_4 * u_5,
            u_6,
            ones,
            u,
            u_2,
            u_3,
            u_4,
            u_5,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        KL = 0.5 * u_K_inv_u + 0.5 * u_K_inv_u2 + 0.5 * u_K_inv_u3 + 0.5 * u_K_inv_u4 + 0.5 * u_K_inv_u5 + 0.5 * u_K_inv_u6
        elbo = -KL - tau * jnp.sum(jnp.square(f_sample.reshape(-1) - self.Y1.reshape(-1))) - tau * jnp.sum(jnp.square(f_sample2.reshape(-1) - self.Y2.reshape(-1))) - tau * jnp.sum(jnp.square(f_sample3.reshape(-1) - self.Y3.reshape(-1))) - tau * jnp.sum(jnp.square(f_sample4.reshape(-1) - self.Y4.reshape(-1))) - tau * jnp.sum(jnp.square(f_sample5.reshape(-1) - self.Y5.reshape(-1))) - tau * jnp.sum(jnp.square(f_sample6.reshape(-1) - self.Y6.reshape(-1))) - co * v * jnp.sum(jnp.square(u_dx1.reshape(-1, 1)[a:b] - jnp.matmul(library1.T, w1.reshape(-1, 1))[a:b])) - co * v * jnp.sum(jnp.square(u_dx1_2.reshape(-1, 1)[a:b] - jnp.matmul(library2.T, w2.reshape(-1, 1))[a:b])) - co * v * jnp.sum(jnp.square(u_dx1_3.reshape(-1, 1)[a:b] - jnp.matmul(library3.T, w3.reshape(-1, 1))[a:b])) - co * v * jnp.sum(jnp.square(u_dx1_4.reshape(-1, 1)[a:b] - jnp.matmul(library4.T, w4.reshape(-1, 1))[a:b])) - co * v * jnp.sum(
            jnp.square(u_dx1_5.reshape(-1, 1)[a:b] - jnp.matmul(library5.T, w5.reshape(-1, 1))[a:b])) - co * v * jnp.sum(jnp.square(u_dx1_6.reshape(-1, 1)[a:b] - jnp.matmul(library6.T, w6.reshape(-1, 1))[a:b]))
        return -elbo.sum()

    @partial(jit, static_argnums=(0, ))
    def pred(self, params_f):
        ls = jnp.exp(params_f['log_ls'])
        ls2 = jnp.exp(params_f['log_ls2'])
        ls3 = jnp.exp(params_f['log_ls3'])
        ls4 = jnp.exp(params_f['log_ls4'])
        ls5 = jnp.exp(params_f['log_ls5'])
        ls6 = jnp.exp(params_f['log_ls6'])
        X_col = self.X_col[0]
        mu = params_f['mu'].sum(axis=-1)
        mu2 = params_f['mu2'].sum(axis=-1)
        mu3 = params_f['mu3'].sum(axis=-1)
        mu4 = params_f['mu4'].sum(axis=-1)
        mu5 = params_f['mu5'].sum(axis=-1)
        mu6 = params_f['mu6'].sum(axis=-1)
        cov_f = self.Kernel.get_cov(self.X_test, X_col, ls)
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        weights = tl.tenalg.multi_mode_dot(mu.reshape(self.num, ), self.K_inv_list)
        pred1 = jnp.matmul(cov_f, weights.reshape(-1, 1))
        cov_f2 = self.Kernel.get_cov(self.X_test, X_col, ls2)
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list2 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list2 = jnp.linalg.inv(self.K_list2)
        weights2 = tl.tenalg.multi_mode_dot(mu2.reshape(self.num, ), self.K_inv_list2)
        pred2 = jnp.matmul(cov_f2, weights2.reshape(-1, 1))
        cov_f3 = self.Kernel.get_cov(self.X_test, X_col, ls3)
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list3 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls3[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list3 = jnp.linalg.inv(self.K_list3)
        weights3 = tl.tenalg.multi_mode_dot(mu3.reshape(self.num, ), self.K_inv_list3)
        pred3 = jnp.matmul(cov_f3, weights3.reshape(-1, 1))
        cov_f4 = self.Kernel.get_cov(self.X_test, X_col, ls4)
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list4 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls4[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list4 = jnp.linalg.inv(self.K_list4)
        weights4 = tl.tenalg.multi_mode_dot(mu4.reshape(self.num, ), self.K_inv_list4)
        pred4 = jnp.matmul(cov_f4, weights4.reshape(-1, 1))
        cov_f5 = self.Kernel.get_cov(self.X_test, X_col, ls5)
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list5 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls5[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list5 = jnp.linalg.inv(self.K_list5)
        weights5 = tl.tenalg.multi_mode_dot(mu5.reshape(self.num, ), self.K_inv_list5)
        pred5 = jnp.matmul(cov_f5, weights5.reshape(-1, 1))
        cov_f6 = self.Kernel.get_cov(self.X_test, X_col, ls6)
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list6 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls6[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list6 = jnp.linalg.inv(self.K_list6)
        weights6 = tl.tenalg.multi_mode_dot(mu6.reshape(self.num, ), self.K_inv_list6)
        pred6 = jnp.matmul(cov_f6, weights6.reshape(-1, 1))
        return pred1.reshape(-1), pred2.reshape(-1), pred3.reshape(-1), pred4.reshape(-1), pred5.reshape(-1), pred6.reshape(-1)

    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, co, w1, w2, w3, w4, w5, w6, tau, v):
        loss, d_params = jax.value_and_grad(self.loss)(params, co, w1, w2, w3, w4, w5, w6, tau, v)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def init_spike_and_slab(self, params):
        mu = params['mu'].sum(axis=-1)
        mu2 = params['mu2'].sum(axis=-1)
        mu3 = params['mu3'].sum(axis=-1)
        mu4 = params['mu4'].sum(axis=-1)
        mu5 = params['mu5'].sum(axis=-1)
        mu6 = params['mu6'].sum(axis=-1)
        u_sample = mu
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        self.derivative_cov_list = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j, :]) for j in range(len(self.derivative_cov_list[0]))] for i in range(len(self.derivative_cov_list))]
        u_sample = mu
        deri_sample = [tl.tenalg.multi_mode_dot(u_sample, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        ls2 = jnp.exp(params['log_ls2'])
        self.Kernel2 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list2 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list2 = jnp.linalg.inv(self.K_list2)
        self.derivative_cov_list2 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls2)
        self.deri_cov_times_K_inv_list2 = [[jnp.matmul(self.derivative_cov_list2[i][j], self.K_inv_list2[j, :]) for j in range(len(self.derivative_cov_list2[0]))] for i in range(len(self.derivative_cov_list2))]
        u_sample2 = mu2
        deri_sample2 = [tl.tenalg.multi_mode_dot(u_sample2, self.deri_cov_times_K_inv_list2[i]) for i in range(len(self.deri_cov_times_K_inv_list2))]
        u_dx1_2 = deri_sample2[1].reshape(1, -1)
        u_2 = deri_sample2[0].reshape(1, -1)
        ls3 = jnp.exp(params['log_ls3'])
        self.Kernel3 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list3 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls3[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list3 = jnp.linalg.inv(self.K_list3)
        self.derivative_cov_list3 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls3)
        self.deri_cov_times_K_inv_list3 = [[jnp.matmul(self.derivative_cov_list3[i][j], self.K_inv_list3[j, :]) for j in range(len(self.derivative_cov_list3[0]))] for i in range(len(self.derivative_cov_list3))]
        deri_sample3 = [tl.tenalg.multi_mode_dot(mu3, self.deri_cov_times_K_inv_list3[i]) for i in range(len(self.deri_cov_times_K_inv_list3))]
        u_dx1_3 = deri_sample3[1].reshape(1, -1)
        u_3 = deri_sample3[0].reshape(1, -1)
        ls4 = jnp.exp(params['log_ls4'])
        self.Kernel4 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list4 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls4[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list4 = jnp.linalg.inv(self.K_list4)
        self.derivative_cov_list4 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls4)
        self.deri_cov_times_K_inv_list4 = [[jnp.matmul(self.derivative_cov_list4[i][j], self.K_inv_list4[j, :]) for j in range(len(self.derivative_cov_list4[0]))] for i in range(len(self.derivative_cov_list4))]
        deri_sample4 = [tl.tenalg.multi_mode_dot(mu4, self.deri_cov_times_K_inv_list4[i]) for i in range(len(self.deri_cov_times_K_inv_list4))]
        u_dx1_4 = deri_sample4[1].reshape(1, -1)
        u_4 = deri_sample4[0].reshape(1, -1)
        ls5 = jnp.exp(params['log_ls5'])
        self.Kernel5 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list5 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls5[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list5 = jnp.linalg.inv(self.K_list5)
        self.derivative_cov_list5 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls5)
        self.deri_cov_times_K_inv_list5 = [[jnp.matmul(self.derivative_cov_list5[i][j], self.K_inv_list5[j, :]) for j in range(len(self.derivative_cov_list5[0]))] for i in range(len(self.derivative_cov_list5))]
        deri_sample5 = [tl.tenalg.multi_mode_dot(mu5, self.deri_cov_times_K_inv_list5[i]) for i in range(len(self.deri_cov_times_K_inv_list5))]
        u_dx1_5 = deri_sample5[1].reshape(1, -1)
        u_5 = deri_sample5[0].reshape(1, -1)
        ls6 = jnp.exp(params['log_ls6'])
        self.Kernel6 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list6 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls6[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list6 = jnp.linalg.inv(self.K_list6)
        self.derivative_cov_list6 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls6)
        self.deri_cov_times_K_inv_list6 = [[jnp.matmul(self.derivative_cov_list6[i][j], self.K_inv_list6[j, :]) for j in range(len(self.derivative_cov_list6[0]))] for i in range(len(self.derivative_cov_list6))]
        deri_sample6 = [tl.tenalg.multi_mode_dot(mu6, self.deri_cov_times_K_inv_list6[i]) for i in range(len(self.deri_cov_times_K_inv_list6))]
        u_dx1_6 = deri_sample6[1].reshape(1, -1)
        u_6 = deri_sample6[0].reshape(1, -1)
        ones = jnp.ones((1, self.num)) * 8
        library1 = jnp.concatenate((
            u_2 * u_6,
            u_5 * u_6,
            u,
            ones,
            u_2,
            u_3,
            u_4,
            u_5,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)

        library2 = jnp.concatenate((
            u * u_3,
            u * u_6,
            u_2,
            ones,
            u,
            u_3,
            u_4,
            u_5,
            u_6,
            u * u_2,
            u * u_4,
            u * u_5,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)

        library3 = jnp.concatenate((
            u_2 * u_4,
            u * u_2,
            u_3,
            ones,
            u,
            u_2,
            u_4,
            u_5,
            u_6,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)

        library4 = jnp.concatenate((
            u_3 * u_5,
            u_2 * u_3,
            u_4,
            ones,
            u,
            u_2,
            u_3,
            u_5,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_6,
            u_4 * u_5,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library5 = jnp.concatenate((
            u_4 * u_6,
            u_3 * u_4,
            u_5,
            ones,
            u,
            u_2,
            u_3,
            u_4,
            u_6,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_5,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_5,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)
        library6 = jnp.concatenate((
            u * u_5,
            u_4 * u_5,
            u_6,
            ones,
            u,
            u_2,
            u_3,
            u_4,
            u_5,
            u * u_2,
            u * u_3,
            u * u_4,
            u * u_6,
            u_2 * u_3,
            u_2 * u_4,
            u_2 * u_5,
            u_2 * u_6,
            u_3 * u_4,
            u_3 * u_5,
            u_3 * u_6,
            u_4 * u_6,
            u_5 * u_6,
            u * u_2**2,
            u * u_3**2,
            u * u_4**2,
            u * u_5**2,
            u * u_6**2,
            u**2 * u_2,
            u_3**2 * u_2,
            u_4**2 * u_2,
            u_5**2 * u_2,
            u_6**2 * u_2,
            u**2 * u_3,
            u_2**2 * u_3,
            u_4**2 * u_3,
            u_5**2 * u_3,
            u_6**2 * u_3,
            u**2 * u_4,
            u_2**2 * u_4,
            u_3**2 * u_4,
            u_5**2 * u_4,
            u_6**2 * u_4,
            u**2 * u_5,
            u_2**2 * u_5,
            u_3**2 * u_5,
            u_4**2 * u_5,
            u_6**2 * u_5,
            u**2 * u_6,
            u_2**2 * u_6,
            u_3**2 * u_6,
            u_4**2 * u_6,
            u_5**2 * u_6,
            u * u_2 * u_3,
            u * u_2 * u_4,
            u * u_2 * u_5,
            u * u_2 * u_6,
            u * u_3 * u_4,
            u * u_3 * u_5,
            u * u_3 * u_6,
            u * u_4 * u_5,
            u * u_4 * u_6,
            u * u_5 * u_6,
            u_2 * u_3 * u_4,
            u_2 * u_3 * u_5,
            u_2 * u_3 * u_6,
            u_2 * u_4 * u_5,
            u_2 * u_4 * u_6,
            u_2 * u_5 * u_6,
            u_3 * u_4 * u_5,
            u_3 * u_4 * u_6,
            u_3 * u_5 * u_6,
            u_4 * u_5 * u_6,
        ), axis=0)

        X1 = np.array(library1.T)
        y1 = np.array(u_dx1.reshape(-1))
        X2 = np.array(library2.T)
        y2 = np.array(u_dx1_2.reshape(-1))

        X3 = np.array(library3.T)
        y3 = np.array(u_dx1_3.reshape(-1))

        X4 = np.array(library4.T)
        y4 = np.array(u_dx1_4.reshape(-1))

        X5 = np.array(library5.T)
        y5 = np.array(u_dx1_5.reshape(-1))

        X6 = np.array(library6.T)
        y6 = np.array(u_dx1_6.reshape(-1))
        
        X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6 = self.get_lib(params)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y4 = np.array(y4)
        y5 = np.array(y5)
        y6 = np.array(y6)
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X4 = np.array(X4)
        X5 = np.array(X5)
        X6 = np.array(X6)
        self.S1 = np.linalg.inv(np.diag(1.0 / self.v_p_full1) + self.tau1 * (X1.T @ X1))
        self.mu_full1 = self.S1 @ (self.mu_p_full1 / self.v_p_full1 + self.tau1 * (X1.T @ y1))
        self.v_full1 = np.diag(self.S1)
        self.S2 = np.linalg.inv(np.diag(1.0 / self.v_p_full2) + self.tau2 * (X2.T @ X2))
        self.mu_full2 = self.S2 @ (self.mu_p_full2 / self.v_p_full2 + self.tau2 * (X2.T @ y2))
        self.v_full2 = np.diag(self.S2)

        self.S3 = np.linalg.inv(np.diag(1.0 / self.v_p_full3) + self.tau3 * (X3.T @ X3))
        self.mu_full3 = self.S3 @ (self.mu_p_full3 / self.v_p_full3 + self.tau3 * (X3.T @ y3))
        self.v_full3 = np.diag(self.S3)
        self.S4 = np.linalg.inv(np.diag(1.0 / self.v_p_full4) + self.tau4 * (X4.T @ X4))
        self.mu_full4 = self.S4 @ (self.mu_p_full4 / self.v_p_full4 + self.tau4 * (X4.T @ y4))
        self.v_full4 = np.diag(self.S4)

        self.S5 = np.linalg.inv(np.diag(1.0 / self.v_p_full5) + self.tau5 * (X5.T @ X5))
        self.mu_full5 = self.S5 @ (self.mu_p_full5 / self.v_p_full5 + self.tau5 * (X5.T @ y5))
        self.v_full5 = np.diag(self.S5)
        self.S6 = np.linalg.inv(np.diag(1.0 / self.v_p_full6) + self.tau6 * (X6.T @ X6))
        self.mu_full6 = self.S6 @ (self.mu_p_full6 / self.v_p_full6 + self.tau6 * (X6.T @ y6))
        self.v_full6 = np.diag(self.S6)

    def spike_and_slab_no_truncate(self, params):
        X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6 = self.get_lib(params)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y4 = np.array(y4)
        y5 = np.array(y5)
        y6 = np.array(y6)
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X4 = np.array(X4)
        X5 = np.array(X5)
        X6 = np.array(X6)

        big_inds1 = np.where(abs(self.mu_full1) > self.bound)[0]
        big_inds2 = np.where(abs(self.mu_full2) > self.bound)[0]
        big_inds3 = np.where(abs(self.mu_full3) > self.bound)[0]
        big_inds4 = np.where(abs(self.mu_full4) > self.bound)[0]
        big_inds5 = np.where(abs(self.mu_full5) > self.bound)[0]
        big_inds6 = np.where(abs(self.mu_full6) > self.bound)[0]
        X1 = X1[:, big_inds1]
        X2 = X2[:, big_inds2]
        X3 = X3[:, big_inds3]
        X4 = X4[:, big_inds4]
        X5 = X5[:, big_inds5]
        X6 = X6[:, big_inds6]
        self.r1 = self.r_full1[big_inds1]
        self.mu1 = self.mu_full1[big_inds1]
        self.v_p1 = self.v_p_full1[big_inds1]
        self.v1 = self.v_full1[big_inds1]
        self.rho_p1 = self.rho_p_full1[big_inds1]
        self.mu_p1 = self.mu_p_full1[big_inds1]

        self.r2 = self.r_full2[big_inds2]
        self.mu2 = self.mu_full2[big_inds2]
        self.v_p2 = self.v_p_full2[big_inds2]
        self.v2 = self.v_full2[big_inds2]
        self.rho_p2 = self.rho_p_full2[big_inds2]
        self.mu_p2 = self.mu_p_full2[big_inds2]

        self.r3 = self.r_full3[big_inds3]
        self.mu3 = self.mu_full3[big_inds3]
        self.v_p3 = self.v_p_full3[big_inds3]
        self.v3 = self.v_full3[big_inds3]
        self.rho_p3 = self.rho_p_full3[big_inds3]
        self.mu_p3 = self.mu_p_full3[big_inds3]

        self.r4 = self.r_full4[big_inds4]
        self.mu4 = self.mu_full4[big_inds4]
        self.v_p4 = self.v_p_full4[big_inds4]
        self.v4 = self.v_full4[big_inds4]
        self.rho_p4 = self.rho_p_full4[big_inds4]
        self.mu_p4 = self.mu_p_full4[big_inds4]

        self.r5 = self.r_full5[big_inds5]
        self.mu5 = self.mu_full5[big_inds5]
        self.v_p5 = self.v_p_full5[big_inds5]
        self.v5 = self.v_full5[big_inds5]
        self.rho_p5 = self.rho_p_full5[big_inds5]
        self.mu_p5 = self.mu_p_full5[big_inds5]

        self.r6 = self.r_full6[big_inds6]
        self.mu6 = self.mu_full6[big_inds6]
        self.v_p6 = self.v_p_full6[big_inds6]
        self.v6 = self.v_full6[big_inds6]
        self.rho_p6 = self.rho_p_full6[big_inds6]
        self.mu_p6 = self.mu_p_full6[big_inds6]
        for i in range(self.num_operators):
            if i not in big_inds1:
                self.mu_full1[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds2:
                self.mu_full2[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds3:
                self.mu_full3[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds4:
                self.mu_full4[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds5:
                self.mu_full5[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds6:
                self.mu_full6[i] = 0
        for it in range(1000):
            old_mu1 = self.mu1.copy()
            v_inv_not1 = 1 / self.v1 - 1 / self.v_p1
            v_not1 = 1 / v_inv_not1
            mu_not1 = v_not1 * (self.mu1 / self.v1 - self.mu_p1 / self.v_p1)
            v_tilt1 = 1 / (1 / v_not1 + 1 / self.s01)
            mu_tilt1 = v_tilt1 * (mu_not1 / v_not1)
            log_h1 = normal.logpdf(mu_not1, scale=np.sqrt(v_not1 + self.s01))
            log_g1 = normal.logpdf(mu_not1, scale=np.sqrt(v_not1))
            rho_p1 = log_h1 - log_g1
            sel_prob1 = expit(self.rho1 + rho_p1)
            mu1 = sel_prob1 * mu_tilt1
            v1 = sel_prob1 * (v_tilt1 + (1.0 - sel_prob1) * mu_tilt1**2)
            self.rho_p1 = self.damping1 * rho_p1 + (1 - self.damping1) * self.rho_p1
            v_p_inv1 = 1 / v1 - v_inv_not1
            v_p_inv1[v_p_inv1 <= 0] = 1 / self.INF1
            v_p_inv_mu1 = mu1 / v1 - mu_not1 / v_not1
            v_p_inv1 = self.damping1 * v_p_inv1 + (1 - self.damping1) * 1 / self.v_p1
            v_p_inv_mu1 = self.damping1 * v_p_inv_mu1 + (1 - self.damping1) * self.mu_p1 / self.v_p1
            self.v_p1 = 1 / v_p_inv1
            self.mu_p1 = self.v_p1 * v_p_inv_mu1
            self.r1 = self.rho_p1 + self.rho1

            self.S1 = np.linalg.inv(np.diag(1.0 / self.v_p1) + self.tau1 * (X1.T @ X1))
            self.mu1 = self.S1 @ (self.mu_p1 / self.v_p1 + self.tau1 * (X1.T @ y1))
            self.v1 = np.diag(self.S1)
            diff1 = np.sum((old_mu1 - self.mu1)**2)
            if diff1 < self.tol1:
                break

        for it in range(1000):
            old_mu2 = self.mu2.copy()
            v_inv_not2 = 1 / self.v2 - 1 / self.v_p2
            v_not2 = 1 / v_inv_not2
            mu_not2 = v_not2 * (self.mu2 / self.v2 - self.mu_p2 / self.v_p2)
            v_tilt2 = 1 / (1 / v_not2 + 1 / self.s02)
            mu_tilt2 = v_tilt2 * (mu_not2 / v_not2)
            log_h2 = normal.logpdf(mu_not2, scale=np.sqrt(v_not2 + self.s02))
            log_g2 = normal.logpdf(mu_not2, scale=np.sqrt(v_not2))
            rho_p2 = log_h2 - log_g2
            sel_prob2 = expit(self.rho2 + rho_p2)
            mu2 = sel_prob2 * mu_tilt2
            v2 = sel_prob2 * (v_tilt2 + (1.0 - sel_prob2) * mu_tilt2**2)
            self.rho_p2 = self.damping2 * rho_p2 + (1 - self.damping2) * self.rho_p2
            v_p_inv2 = 1 / v2 - v_inv_not2
            v_p_inv2[v_p_inv2 <= 0] = 1 / self.INF2
            v_p_inv_mu2 = mu2 / v2 - mu_not2 / v_not2
            v_p_inv2 = self.damping2 * v_p_inv2 + (1 - self.damping2) * 1 / self.v_p2
            v_p_inv_mu2 = self.damping2 * v_p_inv_mu2 + (1 - self.damping2) * self.mu_p2 / self.v_p2
            self.v_p2 = 1 / v_p_inv2
            self.mu_p2 = self.v_p2 * v_p_inv_mu2
            self.r2 = self.rho_p2 + self.rho2
            self.S2 = np.linalg.inv(np.diag(1.0 / self.v_p2) + self.tau2 * (X2.T @ X2))
            self.mu2 = self.S2 @ (self.mu_p2 / self.v_p2 + self.tau2 * (X2.T @ y2))
            self.v2 = np.diag(self.S2)
            diff2 = np.sum((old_mu2 - self.mu2)**2)
            if diff2 < self.tol2:
                break

        for it in range(1000):
            old_mu3 = self.mu3.copy()
            v_inv_not3 = 1 / self.v3 - 1 / self.v_p3
            v_not3 = 1 / v_inv_not3
            mu_not3 = v_not3 * (self.mu3 / self.v3 - self.mu_p3 / self.v_p3)
            v_tilt3 = 1 / (1 / v_not3 + 1 / self.s03)
            mu_tilt3 = v_tilt3 * (mu_not3 / v_not3)
            log_h3 = normal.logpdf(mu_not3, scale=np.sqrt(v_not3 + self.s03))
            log_g3 = normal.logpdf(mu_not3, scale=np.sqrt(v_not3))
            rho_p3 = log_h3 - log_g3
            sel_prob3 = expit(self.rho3 + rho_p3)
            mu3 = sel_prob3 * mu_tilt3
            v3 = sel_prob3 * (v_tilt3 + (1.0 - sel_prob3) * mu_tilt3**2)
            self.rho_p3 = self.damping3 * rho_p3 + (1 - self.damping3) * self.rho_p3
            v_p_inv3 = 1 / v3 - v_inv_not3
            v_p_inv3[v_p_inv3 <= 0] = 1 / self.INF3
            v_p_inv_mu3 = mu3 / v3 - mu_not3 / v_not3
            v_p_inv3 = self.damping3 * v_p_inv3 + (1 - self.damping3) * 1 / self.v_p3
            v_p_inv_mu3 = self.damping3 * v_p_inv_mu3 + (1 - self.damping3) * self.mu_p3 / self.v_p3
            self.v_p3 = 1 / v_p_inv3
            self.mu_p3 = self.v_p3 * v_p_inv_mu3
            self.r3 = self.rho_p3 + self.rho3
            self.S3 = np.linalg.inv(np.diag(1.0 / self.v_p3) + self.tau3 * (X3.T @ X3))
            self.mu3 = self.S3 @ (self.mu_p3 / self.v_p3 + self.tau3 * (X3.T @ y3))
            self.v3 = np.diag(self.S3)
            diff3 = np.sum((old_mu3 - self.mu3)**2)
            if diff3 < self.tol3:
                break

        for it in range(1000):
            old_mu4 = self.mu4.copy()
            v_inv_not4 = 1 / self.v4 - 1 / self.v_p4
            v_not4 = 1 / v_inv_not4
            mu_not4 = v_not4 * (self.mu4 / self.v4 - self.mu_p4 / self.v_p4)
            v_tilt4 = 1 / (1 / v_not4 + 1 / self.s04)
            mu_tilt4 = v_tilt4 * (mu_not4 / v_not4)
            log_h4 = normal.logpdf(mu_not4, scale=np.sqrt(v_not4 + self.s04))
            log_g4 = normal.logpdf(mu_not4, scale=np.sqrt(v_not4))
            rho_p4 = log_h4 - log_g4
            sel_prob4 = expit(self.rho4 + rho_p4)
            mu4 = sel_prob4 * mu_tilt4
            v4 = sel_prob4 * (v_tilt4 + (1.0 - sel_prob4) * mu_tilt4**2)
            self.rho_p4 = self.damping4 * rho_p4 + (1 - self.damping4) * self.rho_p4
            v_p_inv4 = 1 / v4 - v_inv_not4
            v_p_inv4[v_p_inv4 <= 0] = 1 / self.INF4
            v_p_inv_mu4 = mu4 / v4 - mu_not4 / v_not4
            v_p_inv4 = self.damping4 * v_p_inv4 + (1 - self.damping4) * 1 / self.v_p4
            v_p_inv_mu4 = self.damping4 * v_p_inv_mu4 + (1 - self.damping4) * self.mu_p4 / self.v_p4
            self.v_p4 = 1 / v_p_inv4
            self.mu_p4 = self.v_p4 * v_p_inv_mu4
            self.r4 = self.rho_p4 + self.rho4
            self.S4 = np.linalg.inv(np.diag(1.0 / self.v_p4) + self.tau4 * (X4.T @ X4))
            self.mu4 = self.S4 @ (self.mu_p4 / self.v_p4 + self.tau4 * (X4.T @ y4))
            self.v4 = np.diag(self.S4)
            diff4 = np.sum((old_mu4 - self.mu4)**2)
            if diff4 < self.tol4:
                break

        for it in range(1000):
            old_mu5 = self.mu5.copy()
            v_inv_not5 = 1 / self.v5 - 1 / self.v_p5
            v_not5 = 1 / v_inv_not5
            mu_not5 = v_not5 * (self.mu5 / self.v5 - self.mu_p5 / self.v_p5)
            v_tilt5 = 1 / (1 / v_not5 + 1 / self.s05)
            mu_tilt5 = v_tilt5 * (mu_not5 / v_not5)
            log_h5 = normal.logpdf(mu_not5, scale=np.sqrt(v_not5 + self.s05))
            log_g5 = normal.logpdf(mu_not5, scale=np.sqrt(v_not5))
            rho_p5 = log_h5 - log_g5
            sel_prob5 = expit(self.rho5 + rho_p5)
            mu5 = sel_prob5 * mu_tilt5
            v5 = sel_prob5 * (v_tilt5 + (1.0 - sel_prob5) * mu_tilt5**2)
            self.rho_p5 = self.damping5 * rho_p5 + (1 - self.damping5) * self.rho_p5
            v_p_inv5 = 1 / v5 - v_inv_not5
            v_p_inv5[v_p_inv5 <= 0] = 1 / self.INF5
            v_p_inv_mu5 = mu5 / v5 - mu_not5 / v_not5
            v_p_inv5 = self.damping5 * v_p_inv5 + (1 - self.damping5) * 1 / self.v_p5
            v_p_inv_mu5 = self.damping5 * v_p_inv_mu5 + (1 - self.damping5) * self.mu_p5 / self.v_p5
            self.v_p5 = 1 / v_p_inv5
            self.mu_p5 = self.v_p5 * v_p_inv_mu5
            self.r5 = self.rho_p5 + self.rho5
            self.S5 = np.linalg.inv(np.diag(1.0 / self.v_p5) + self.tau5 * (X5.T @ X5))
            self.mu5 = self.S5 @ (self.mu_p5 / self.v_p5 + self.tau5 * (X5.T @ y5))
            self.v5 = np.diag(self.S5)
            diff5 = np.sum((old_mu5 - self.mu5)**2)
            if diff5 < self.tol5:
                break

        for it in range(1000):
            old_mu6 = self.mu6.copy()
            v_inv_not6 = 1 / self.v6 - 1 / self.v_p6
            v_not6 = 1 / v_inv_not6
            mu_not6 = v_not6 * (self.mu6 / self.v6 - self.mu_p6 / self.v_p6)
            v_tilt6 = 1 / (1 / v_not6 + 1 / self.s06)
            mu_tilt6 = v_tilt6 * (mu_not6 / v_not6)
            log_h6 = normal.logpdf(mu_not6, scale=np.sqrt(v_not6 + self.s06))
            log_g6 = normal.logpdf(mu_not6, scale=np.sqrt(v_not6))
            rho_p6 = log_h6 - log_g6
            sel_prob6 = expit(self.rho6 + rho_p6)
            mu6 = sel_prob6 * mu_tilt6
            v6 = sel_prob6 * (v_tilt6 + (1.0 - sel_prob6) * mu_tilt6**2)
            self.rho_p6 = self.damping6 * rho_p6 + (1 - self.damping6) * self.rho_p6
            v_p_inv6 = 1 / v6 - v_inv_not6
            v_p_inv6[v_p_inv6 <= 0] = 1 / self.INF6
            v_p_inv_mu6 = mu6 / v6 - mu_not6 / v_not6
            v_p_inv6 = self.damping6 * v_p_inv6 + (1 - self.damping6) * 1 / self.v_p6
            v_p_inv_mu6 = self.damping6 * v_p_inv_mu6 + (1 - self.damping6) * self.mu_p6 / self.v_p6
            self.v_p6 = 1 / v_p_inv6
            self.mu_p6 = self.v_p6 * v_p_inv_mu6
            self.r6 = self.rho_p6 + self.rho6
            self.S6 = np.linalg.inv(np.diag(1.0 / self.v_p6) + self.tau6 * (X6.T @ X6))
            self.mu6 = self.S6 @ (self.mu_p6 / self.v_p6 + self.tau6 * (X6.T @ y6))
            self.v6 = np.diag(self.S6)
            diff6 = np.sum((old_mu6 - self.mu6)**2)
            if diff6 < self.tol6:
                break

        self.sel_prob1 = expit(self.r1)
        self.sel_prob2 = expit(self.r2)
        self.sel_prob3 = expit(self.r3)
        self.sel_prob4 = expit(self.r4)
        self.sel_prob5 = expit(self.r5)
        self.sel_prob6 = expit(self.r6)
        self.r_full1[big_inds1] = self.r1
        self.mu_full1[big_inds1] = self.mu1
        self.v_p_full1[big_inds1] = self.v_p1
        self.v_full1 = self.v_full1.copy()
        self.v_full1[big_inds1] = self.v1
        self.rho_p_full1[big_inds1] = self.rho_p1
        self.mu_p_full1[big_inds1] = self.mu_p1
        ind1 = 0
        new_mu1 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds1:
                new_mu1[i] = self.mu1[ind1]
                ind1 = ind1 + 1
            else:
                self.mu_full1[i] = 0

        self.r_full2[big_inds2] = self.r2
        self.mu_full2[big_inds2] = self.mu2
        self.v_p_full2[big_inds2] = self.v_p2
        self.v_full2 = self.v_full2.copy()
        self.v_full2[big_inds2] = self.v2
        self.rho_p_full2[big_inds2] = self.rho_p2
        self.mu_p_full2[big_inds2] = self.mu_p2
        ind2 = 0
        new_mu2 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds2:
                new_mu2[i] = self.mu2[ind2]
                ind2 = ind2 + 1
            else:
                self.mu_full2[i] = 0

        self.r_full3[big_inds3] = self.r3
        self.mu_full3[big_inds3] = self.mu3
        self.v_p_full3[big_inds3] = self.v_p3
        self.v_full3 = self.v_full3.copy()
        self.v_full3[big_inds3] = self.v3
        self.rho_p_full3[big_inds3] = self.rho_p3
        self.mu_p_full3[big_inds3] = self.mu_p3
        ind3 = 0
        new_mu3 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds3:
                new_mu3[i] = self.mu3[ind3]
                ind3 = ind3 + 1
            else:
                self.mu_full3[i] = 0

        self.r_full4[big_inds4] = self.r4
        self.mu_full4[big_inds4] = self.mu4
        self.v_p_full4[big_inds4] = self.v_p4
        self.v_full4 = self.v_full4.copy()
        self.v_full4[big_inds4] = self.v4
        self.rho_p_full4[big_inds4] = self.rho_p4
        self.mu_p_full4[big_inds4] = self.mu_p4
        ind4 = 0
        new_mu4 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds4:
                new_mu4[i] = self.mu4[ind4]
                ind4 = ind4 + 1
            else:
                self.mu_full4[i] = 0

        self.r_full5[big_inds5] = self.r5
        self.mu_full5[big_inds5] = self.mu5
        self.v_p_full5[big_inds5] = self.v_p5
        self.v_full5 = self.v_full5.copy()
        self.v_full5[big_inds5] = self.v5
        self.rho_p_full5[big_inds5] = self.rho_p5
        self.mu_p_full5[big_inds5] = self.mu_p5
        ind5 = 0
        new_mu5 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds5:
                new_mu5[i] = self.mu5[ind5]
                ind5 = ind5 + 1
            else:
                self.mu_full5[i] = 0

        self.r_full6[big_inds6] = self.r6
        self.mu_full6[big_inds6] = self.mu6
        self.v_p_full6[big_inds6] = self.v_p6
        self.v_full6 = self.v_full6.copy()
        self.v_full6[big_inds6] = self.v6
        self.rho_p_full6[big_inds6] = self.rho_p6
        self.mu_p_full6[big_inds6] = self.mu_p6
        ind6 = 0
        new_mu6 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds6:
                new_mu6[i] = self.mu6[ind6]
                ind6 = ind6 + 1
            else:
                self.mu_full6[i] = 0
        return new_mu1.reshape(-1), new_mu2.reshape(-1), new_mu3.reshape(-1), new_mu4.reshape(-1), new_mu5.reshape(-1), new_mu6.reshape(-1)

    def spike_and_slab_truncate(self, params):
        X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6 = self.get_lib(params)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y4 = np.array(y4)
        y5 = np.array(y5)
        y6 = np.array(y6)
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X4 = np.array(X4)
        X5 = np.array(X5)
        X6 = np.array(X6)
        big_inds1 = np.where(abs(self.mu_full1) > self.bound)[0]
        big_inds2 = np.where(abs(self.mu_full2) > self.bound)[0]
        big_inds3 = np.where(abs(self.mu_full3) > self.bound)[0]
        big_inds4 = np.where(abs(self.mu_full4) > self.bound)[0]
        big_inds5 = np.where(abs(self.mu_full5) > self.bound)[0]
        big_inds6 = np.where(abs(self.mu_full6) > self.bound)[0]
        X1 = X1[:, big_inds1]
        X2 = X2[:, big_inds2]
        X3 = X3[:, big_inds3]
        X4 = X4[:, big_inds4]
        X5 = X5[:, big_inds5]
        X6 = X6[:, big_inds6]
        self.r1 = self.r_full1[big_inds1]
        self.mu1 = self.mu_full1[big_inds1]
        self.v_p1 = self.v_p_full1[big_inds1]
        self.v1 = self.v_full1[big_inds1]
        self.rho_p1 = self.rho_p_full1[big_inds1]
        self.mu_p1 = self.mu_p_full1[big_inds1]

        self.r2 = self.r_full2[big_inds2]
        self.mu2 = self.mu_full2[big_inds2]
        self.v_p2 = self.v_p_full2[big_inds2]
        self.v2 = self.v_full2[big_inds2]
        self.rho_p2 = self.rho_p_full2[big_inds2]
        self.mu_p2 = self.mu_p_full2[big_inds2]

        self.r3 = self.r_full3[big_inds3]
        self.mu3 = self.mu_full3[big_inds3]
        self.v_p3 = self.v_p_full3[big_inds3]
        self.v3 = self.v_full3[big_inds3]
        self.rho_p3 = self.rho_p_full3[big_inds3]
        self.mu_p3 = self.mu_p_full3[big_inds3]

        self.r4 = self.r_full4[big_inds4]
        self.mu4 = self.mu_full4[big_inds4]
        self.v_p4 = self.v_p_full4[big_inds4]
        self.v4 = self.v_full4[big_inds4]
        self.rho_p4 = self.rho_p_full4[big_inds4]
        self.mu_p4 = self.mu_p_full4[big_inds4]

        self.r5 = self.r_full5[big_inds5]
        self.mu5 = self.mu_full5[big_inds5]
        self.v_p5 = self.v_p_full5[big_inds5]
        self.v5 = self.v_full5[big_inds5]
        self.rho_p5 = self.rho_p_full5[big_inds5]
        self.mu_p5 = self.mu_p_full5[big_inds5]

        self.r6 = self.r_full6[big_inds6]
        self.mu6 = self.mu_full6[big_inds6]
        self.v_p6 = self.v_p_full6[big_inds6]
        self.v6 = self.v_full6[big_inds6]
        self.rho_p6 = self.rho_p_full6[big_inds6]
        self.mu_p6 = self.mu_p_full6[big_inds6]
        for i in range(self.num_operators):
            if i not in big_inds1:
                self.mu_full1[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds2:
                self.mu_full2[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds3:
                self.mu_full3[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds4:
                self.mu_full4[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds5:
                self.mu_full5[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds6:
                self.mu_full6[i] = 0
        for it in range(1000):
            old_mu1 = self.mu1.copy()
            v_inv_not1 = 1 / self.v1 - 1 / self.v_p1
            v_not1 = 1 / v_inv_not1
            mu_not1 = v_not1 * (self.mu1 / self.v1 - self.mu_p1 / self.v_p1)
            v_tilt1 = 1 / (1 / v_not1 + 1 / self.s01)
            mu_tilt1 = v_tilt1 * (mu_not1 / v_not1)
            log_h1 = normal.logpdf(mu_not1, scale=np.sqrt(v_not1 + self.s01))
            log_g1 = normal.logpdf(mu_not1, scale=np.sqrt(v_not1))
            rho_p1 = log_h1 - log_g1
            sel_prob1 = expit(self.rho1 + rho_p1)
            mu1 = sel_prob1 * mu_tilt1
            v1 = sel_prob1 * (v_tilt1 + (1.0 - sel_prob1) * mu_tilt1**2)
            self.rho_p1 = self.damping1 * rho_p1 + (1 - self.damping1) * self.rho_p1
            v_p_inv1 = 1 / v1 - v_inv_not1
            v_p_inv1[v_p_inv1 <= 0] = 1 / self.INF1
            v_p_inv_mu1 = mu1 / v1 - mu_not1 / v_not1
            v_p_inv1 = self.damping1 * v_p_inv1 + (1 - self.damping1) * 1 / self.v_p1
            v_p_inv_mu1 = self.damping1 * v_p_inv_mu1 + (1 - self.damping1) * self.mu_p1 / self.v_p1
            self.v_p1 = 1 / v_p_inv1
            self.mu_p1 = self.v_p1 * v_p_inv_mu1
            self.r1 = self.rho_p1 + self.rho1
            self.S1 = np.linalg.inv(np.diag(1.0 / self.v_p1) + self.tau1 * (X1.T @ X1))
            self.mu1 = self.S1 @ (self.mu_p1 / self.v_p1 + self.tau1 * (X1.T @ y1))
            self.v1 = np.diag(self.S1)
            diff1 = np.sum((old_mu1 - self.mu1)**2)
            if diff1 < self.tol1:
                break

        for it in range(1000):
            old_mu2 = self.mu2.copy()
            v_inv_not2 = 1 / self.v2 - 1 / self.v_p2
            v_not2 = 1 / v_inv_not2
            mu_not2 = v_not2 * (self.mu2 / self.v2 - self.mu_p2 / self.v_p2)
            v_tilt2 = 1 / (1 / v_not2 + 1 / self.s02)
            mu_tilt2 = v_tilt2 * (mu_not2 / v_not2)
            log_h2 = normal.logpdf(mu_not2, scale=np.sqrt(v_not2 + self.s02))
            log_g2 = normal.logpdf(mu_not2, scale=np.sqrt(v_not2))
            rho_p2 = log_h2 - log_g2
            sel_prob2 = expit(self.rho2 + rho_p2)
            mu2 = sel_prob2 * mu_tilt2
            v2 = sel_prob2 * (v_tilt2 + (1.0 - sel_prob2) * mu_tilt2**2)
            self.rho_p2 = self.damping2 * rho_p2 + (1 - self.damping2) * self.rho_p2
            v_p_inv2 = 1 / v2 - v_inv_not2
            v_p_inv2[v_p_inv2 <= 0] = 1 / self.INF2
            v_p_inv_mu2 = mu2 / v2 - mu_not2 / v_not2
            v_p_inv2 = self.damping2 * v_p_inv2 + (1 - self.damping2) * 1 / self.v_p2
            v_p_inv_mu2 = self.damping2 * v_p_inv_mu2 + (1 - self.damping2) * self.mu_p2 / self.v_p2
            self.v_p2 = 1 / v_p_inv2
            self.mu_p2 = self.v_p2 * v_p_inv_mu2
            self.r2 = self.rho_p2 + self.rho2
            self.S2 = np.linalg.inv(np.diag(1.0 / self.v_p2) + self.tau2 * (X2.T @ X2))
            self.mu2 = self.S2 @ (self.mu_p2 / self.v_p2 + self.tau2 * (X2.T @ y2))
            self.v2 = np.diag(self.S2)
            diff2 = np.sum((old_mu2 - self.mu2)**2)
            if diff2 < self.tol2:
                break

        for it in range(1000):
            old_mu3 = self.mu3.copy()
            v_inv_not3 = 1 / self.v3 - 1 / self.v_p3
            v_not3 = 1 / v_inv_not3
            mu_not3 = v_not3 * (self.mu3 / self.v3 - self.mu_p3 / self.v_p3)
            v_tilt3 = 1 / (1 / v_not3 + 1 / self.s03)
            mu_tilt3 = v_tilt3 * (mu_not3 / v_not3)
            log_h3 = normal.logpdf(mu_not3, scale=np.sqrt(v_not3 + self.s03))
            log_g3 = normal.logpdf(mu_not3, scale=np.sqrt(v_not3))
            rho_p3 = log_h3 - log_g3
            sel_prob3 = expit(self.rho3 + rho_p3)
            mu3 = sel_prob3 * mu_tilt3
            v3 = sel_prob3 * (v_tilt3 + (1.0 - sel_prob3) * mu_tilt3**2)
            self.rho_p3 = self.damping3 * rho_p3 + (1 - self.damping3) * self.rho_p3
            v_p_inv3 = 1 / v3 - v_inv_not3
            v_p_inv3[v_p_inv3 <= 0] = 1 / self.INF3
            v_p_inv_mu3 = mu3 / v3 - mu_not3 / v_not3
            v_p_inv3 = self.damping3 * v_p_inv3 + (1 - self.damping3) * 1 / self.v_p3
            v_p_inv_mu3 = self.damping3 * v_p_inv_mu3 + (1 - self.damping3) * self.mu_p3 / self.v_p3
            self.v_p3 = 1 / v_p_inv3
            self.mu_p3 = self.v_p3 * v_p_inv_mu3
            self.r3 = self.rho_p3 + self.rho3
            self.S3 = np.linalg.inv(np.diag(1.0 / self.v_p3) + self.tau3 * (X3.T @ X3))
            self.mu3 = self.S3 @ (self.mu_p3 / self.v_p3 + self.tau3 * (X3.T @ y3))
            self.v3 = np.diag(self.S3)
            diff3 = np.sum((old_mu3 - self.mu3)**2)
            if diff3 < self.tol3:
                break

        for it in range(1000):
            old_mu4 = self.mu4.copy()
            v_inv_not4 = 1 / self.v4 - 1 / self.v_p4
            v_not4 = 1 / v_inv_not4
            mu_not4 = v_not4 * (self.mu4 / self.v4 - self.mu_p4 / self.v_p4)
            v_tilt4 = 1 / (1 / v_not4 + 1 / self.s04)
            mu_tilt4 = v_tilt4 * (mu_not4 / v_not4)
            log_h4 = normal.logpdf(mu_not4, scale=np.sqrt(v_not4 + self.s04))
            log_g4 = normal.logpdf(mu_not4, scale=np.sqrt(v_not4))
            rho_p4 = log_h4 - log_g4
            sel_prob4 = expit(self.rho4 + rho_p4)
            mu4 = sel_prob4 * mu_tilt4
            v4 = sel_prob4 * (v_tilt4 + (1.0 - sel_prob4) * mu_tilt4**2)
            self.rho_p4 = self.damping4 * rho_p4 + (1 - self.damping4) * self.rho_p4
            v_p_inv4 = 1 / v4 - v_inv_not4
            v_p_inv4[v_p_inv4 <= 0] = 1 / self.INF4
            v_p_inv_mu4 = mu4 / v4 - mu_not4 / v_not4
            v_p_inv4 = self.damping4 * v_p_inv4 + (1 - self.damping4) * 1 / self.v_p4
            v_p_inv_mu4 = self.damping4 * v_p_inv_mu4 + (1 - self.damping4) * self.mu_p4 / self.v_p4
            self.v_p4 = 1 / v_p_inv4
            self.mu_p4 = self.v_p4 * v_p_inv_mu4
            self.r4 = self.rho_p4 + self.rho4
            self.S4 = np.linalg.inv(np.diag(1.0 / self.v_p4) + self.tau4 * (X4.T @ X4))
            self.mu4 = self.S4 @ (self.mu_p4 / self.v_p4 + self.tau4 * (X4.T @ y4))
            self.v4 = np.diag(self.S4)
            diff4 = np.sum((old_mu4 - self.mu4)**2)
            if diff4 < self.tol4:
                break

        for it in range(1000):
            old_mu5 = self.mu5.copy()
            v_inv_not5 = 1 / self.v5 - 1 / self.v_p5
            v_not5 = 1 / v_inv_not5
            mu_not5 = v_not5 * (self.mu5 / self.v5 - self.mu_p5 / self.v_p5)
            v_tilt5 = 1 / (1 / v_not5 + 1 / self.s05)
            mu_tilt5 = v_tilt5 * (mu_not5 / v_not5)
            log_h5 = normal.logpdf(mu_not5, scale=np.sqrt(v_not5 + self.s05))
            log_g5 = normal.logpdf(mu_not5, scale=np.sqrt(v_not5))
            rho_p5 = log_h5 - log_g5
            sel_prob5 = expit(self.rho5 + rho_p5)
            mu5 = sel_prob5 * mu_tilt5
            v5 = sel_prob5 * (v_tilt5 + (1.0 - sel_prob5) * mu_tilt5**2)
            self.rho_p5 = self.damping5 * rho_p5 + (1 - self.damping5) * self.rho_p5
            v_p_inv5 = 1 / v5 - v_inv_not5
            v_p_inv5[v_p_inv5 <= 0] = 1 / self.INF5
            v_p_inv_mu5 = mu5 / v5 - mu_not5 / v_not5
            v_p_inv5 = self.damping5 * v_p_inv5 + (1 - self.damping5) * 1 / self.v_p5
            v_p_inv_mu5 = self.damping5 * v_p_inv_mu5 + (1 - self.damping5) * self.mu_p5 / self.v_p5
            self.v_p5 = 1 / v_p_inv5
            self.mu_p5 = self.v_p5 * v_p_inv_mu5
            self.r5 = self.rho_p5 + self.rho5
            self.S5 = np.linalg.inv(np.diag(1.0 / self.v_p5) + self.tau5 * (X5.T @ X5))
            self.mu5 = self.S5 @ (self.mu_p5 / self.v_p5 + self.tau5 * (X5.T @ y5))
            self.v5 = np.diag(self.S5)
            diff5 = np.sum((old_mu5 - self.mu5)**2)
            if diff5 < self.tol5:
                break

        for it in range(1000):
            old_mu6 = self.mu6.copy()
            v_inv_not6 = 1 / self.v6 - 1 / self.v_p6
            v_not6 = 1 / v_inv_not6
            mu_not6 = v_not6 * (self.mu6 / self.v6 - self.mu_p6 / self.v_p6)
            v_tilt6 = 1 / (1 / v_not6 + 1 / self.s06)
            mu_tilt6 = v_tilt6 * (mu_not6 / v_not6)
            log_h6 = normal.logpdf(mu_not6, scale=np.sqrt(v_not6 + self.s06))
            log_g6 = normal.logpdf(mu_not6, scale=np.sqrt(v_not6))
            rho_p6 = log_h6 - log_g6
            sel_prob6 = expit(self.rho6 + rho_p6)
            mu6 = sel_prob6 * mu_tilt6
            v6 = sel_prob6 * (v_tilt6 + (1.0 - sel_prob6) * mu_tilt6**2)
            self.rho_p6 = self.damping6 * rho_p6 + (1 - self.damping6) * self.rho_p6
            v_p_inv6 = 1 / v6 - v_inv_not6
            v_p_inv6[v_p_inv6 <= 0] = 1 / self.INF6
            v_p_inv_mu6 = mu6 / v6 - mu_not6 / v_not6
            v_p_inv6 = self.damping6 * v_p_inv6 + (1 - self.damping6) * 1 / self.v_p6
            v_p_inv_mu6 = self.damping6 * v_p_inv_mu6 + (1 - self.damping6) * self.mu_p6 / self.v_p6
            self.v_p6 = 1 / v_p_inv6
            self.mu_p6 = self.v_p6 * v_p_inv_mu6
            self.r6 = self.rho_p6 + self.rho6
            self.S6 = np.linalg.inv(np.diag(1.0 / self.v_p6) + self.tau6 * (X6.T @ X6))
            self.mu6 = self.S6 @ (self.mu_p6 / self.v_p6 + self.tau6 * (X6.T @ y6))
            self.v6 = np.diag(self.S6)
            diff6 = np.sum((old_mu6 - self.mu6)**2)
            if diff6 < self.tol6:
                break

        self.sel_prob1 = expit(self.r1)
        self.sel_prob2 = expit(self.r2)
        self.sel_prob3 = expit(self.r3)
        self.sel_prob4 = expit(self.r4)
        self.sel_prob5 = expit(self.r5)
        self.sel_prob6 = expit(self.r6)
        # print(self.sel_prob1[:10])
        # print(self.sel_prob2[:10])
        # print(self.sel_prob3[:10])
        # print(self.sel_prob4[:10])
        # print(self.sel_prob5[:10])
        # print(self.sel_prob6[:10])
        self.r_full1[big_inds1] = self.r1
        smallest1 = big_inds1[np.argmax(-self.r1)]
        if self.sel_prob1[np.argmax(-self.r1)] > self.th:
            smallest1 = -1
        self.mu_full1[big_inds1] = self.mu1
        self.v_p_full1[big_inds1] = self.v_p1
        self.v_full1 = self.v_full1.copy()
        self.v_full1[big_inds1] = self.v1
        self.rho_p_full1[big_inds1] = self.rho_p1
        self.mu_p_full1[big_inds1] = self.mu_p1
        ind1 = 0
        new_mu1 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds1:
                if i != smallest1:
                    new_mu1[i] = self.mu1[ind1]
                else:
                    if self.r1.shape[0] >= 1:
                        self.mu_full1[i] = 0
                ind1 = ind1 + 1
            else:
                self.mu_full1[i] = 0
        self.r_full2[big_inds2] = self.r2
        smallest2 = big_inds2[np.argmax(-self.r2)]
        if self.sel_prob2[np.argmax(-self.r2)] > self.th:
            smallest2 = -1
        self.mu_full2[big_inds2] = self.mu2
        self.v_p_full2[big_inds2] = self.v_p2
        self.v_full2 = self.v_full2.copy()
        self.v_full2[big_inds2] = self.v2
        self.rho_p_full2[big_inds2] = self.rho_p2
        self.mu_p_full2[big_inds2] = self.mu_p2
        ind2 = 0
        new_mu2 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds2:
                if i != smallest2:
                    new_mu2[i] = self.mu2[ind2]
                else:
                    if self.r2.shape[0] >= 1:
                        self.mu_full2[i] = 0
                ind2 = ind2 + 1
            else:
                self.mu_full2[i] = 0
        self.r_full3[big_inds3] = self.r3
        smallest3 = big_inds3[np.argmax(-self.r3)]
        if self.sel_prob3[np.argmax(-self.r3)] > self.th:
            smallest3 = -1
        self.mu_full3[big_inds3] = self.mu3
        self.v_p_full3[big_inds3] = self.v_p3
        self.v_full3 = self.v_full3.copy()
        self.v_full3[big_inds3] = self.v3
        self.rho_p_full3[big_inds3] = self.rho_p3
        self.mu_p_full3[big_inds3] = self.mu_p3
        ind3 = 0
        new_mu3 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds3:
                if i != smallest3:
                    new_mu3[i] = self.mu3[ind3]
                else:
                    if self.r3.shape[0] >= 1:
                        self.mu_full3[i] = 0
                ind3 = ind3 + 1
            else:
                self.mu_full3[i] = 0
        self.r_full4[big_inds4] = self.r4
        smallest4 = big_inds4[np.argmax(-self.r4)]
        if self.sel_prob4[np.argmax(-self.r4)] > self.th:
            smallest4 = -1
        self.mu_full4[big_inds4] = self.mu4
        self.v_p_full4[big_inds4] = self.v_p4
        self.v_full4 = self.v_full4.copy()
        self.v_full4[big_inds4] = self.v4
        self.rho_p_full4[big_inds4] = self.rho_p4
        self.mu_p_full4[big_inds4] = self.mu_p4
        ind4 = 0
        new_mu4 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds4:
                if i != smallest4:
                    new_mu4[i] = self.mu4[ind4]
                else:
                    if self.r4.shape[0] >= 1:
                        self.mu_full4[i] = 0
                ind4 = ind4 + 1
            else:
                self.mu_full4[i] = 0
        self.r_full5[big_inds5] = self.r5
        smallest5 = big_inds5[np.argmax(-self.r5)]
        if self.sel_prob5[np.argmax(-self.r5)] > self.th:
            smallest5 = -1
        self.mu_full5[big_inds5] = self.mu5
        self.v_p_full5[big_inds5] = self.v_p5
        self.v_full5 = self.v_full5.copy()
        self.v_full5[big_inds5] = self.v5
        self.rho_p_full5[big_inds5] = self.rho_p5
        self.mu_p_full5[big_inds5] = self.mu_p5
        ind5 = 0
        new_mu5 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds5:
                if i != smallest5:
                    new_mu5[i] = self.mu5[ind5]
                else:
                    if self.r5.shape[0] >= 1:
                        self.mu_full5[i] = 0
                ind5 = ind5 + 1
            else:
                self.mu_full5[i] = 0
        self.r_full6[big_inds6] = self.r6
        smallest6 = big_inds6[np.argmax(-self.r6)]
        if self.sel_prob6[np.argmax(-self.r6)] > self.th:
            smallest6 = -1
        self.mu_full6[big_inds6] = self.mu6
        self.v_p_full6[big_inds6] = self.v_p6
        self.v_full6 = self.v_full6.copy()
        self.v_full6[big_inds6] = self.v6
        self.rho_p_full6[big_inds6] = self.rho_p6
        self.mu_p_full6[big_inds6] = self.mu_p6
        ind6 = 0
        new_mu6 = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds6:
                if i != smallest6:
                    new_mu6[i] = self.mu6[ind6]
                else:
                    if self.r6.shape[0] >= 1:
                        self.mu_full6[i] = 0
                ind6 = ind6 + 1
            else:
                self.mu_full6[i] = 0
        return new_mu1.reshape(-1), new_mu2.reshape(-1), new_mu3.reshape(-1), new_mu4.reshape(-1), new_mu5.reshape(-1), new_mu6.reshape(-1)

    def train(self):
        key = jax.random.PRNGKey(0)
        params_f = {
            'mu': np.zeros((self.num, 100)),
            'log_ls': np.array([-0.2]),
            'mu2': np.zeros((self.num, 100)),
            'log_ls2': np.array([-0.2]),
            'mu3': np.zeros((self.num, 100)),
            'log_ls3': np.array([-0.2]),
            'mu4': np.zeros((self.num, 100)),
            'log_ls4': np.array([-0.2]),
            'mu5': np.zeros((self.num, 100)),
            'log_ls5': np.array([-0.2]),
            'mu6': np.zeros((self.num, 100)),
            'log_ls6': np.array([-0.2]),
        }
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params_f)
        co = 0.0
        w1 = jnp.zeros(self.num_operators)
        w2 = jnp.zeros(self.num_operators)
        w3 = jnp.zeros(self.num_operators)
        w4 = jnp.zeros(self.num_operators)
        w5 = jnp.zeros(self.num_operators)
        w6 = jnp.zeros(self.num_operators)
        start_EQD = 3200
        EQD_interval = 200
        for i in range(50000):
            params_f, opt_state, _ = self.step(optimizer, params_f, opt_state, co, w1, w2, w3, w4, w5, w6, 20000, 10)
            if i + 1 == start_EQD:
                self.init_spike_and_slab(params_f)
                w1, w2, w3, w4, w5, w6 = self.spike_and_slab_truncate(params_f)
                w1, w2, w3, w4, w5, w6 = self.spike_and_slab_no_truncate(params_f)
                co = 1.0
                optimizer = optax.adam(1e-4)
                opt_state = optimizer.init(params_f)
            if i + 1 == 20000:
                self.bound = 0.1
                self.tau1 = 100
                self.tau2 = 100
                self.tau3 = 100
                self.tau4 = 100
                self.tau5 = 100
                self.tau6 = 100
            if (i + 1) % 100 == 0 and (i + 1) > start_EQD:
                print(i," Learned ", w1, "\n", w2, '\n', w3, '\n', w4, '\n', w5, '\n', w6)
            if (i + 1) % EQD_interval == 0 and (i + 1) > start_EQD:
                w1, w2, w3, w4, w5, w6 = self.spike_and_slab_truncate(params_f)
                w1, w2, w3, w4, w5, w6 = self.spike_and_slab_no_truncate(params_f)


X_train = np.load('xtrain_lo.npy')
Y_train = np.load('ytrain_lo.npy')
X_test = np.load('xtest_lo.npy')
Y_test = np.load('ytest_lo.npy')
sensors = [X_train.reshape(-1)]
MEQD = EQD(sensors, Y_train, X_test, Y_test)
MEQD.train()
