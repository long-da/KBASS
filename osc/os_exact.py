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

np.random.seed(0)
random.seed(0)
print("Jax on", xla_bridge.get_backend().platform)
tl.set_backend('jax')

class EQD:

    def __init__(self, sensors, Y_train, X_test, Y_test,X_train):
        self.sensors = sensors
        self.X=X_train
        self.Y1 = Y_train[:, 0].reshape(-1, 1)
        self.Y2 = Y_train[:, 1].reshape(-1, 1)
        self.jitter = 1e-2
        self.num = 200
        self.d = 1
        self.X_test = X_test
        self.Y_test = Y_test
        self.lb = X_test.min(axis=0)
        self.ub = X_test.max(axis=0)
        self.X_col = np.array([np.linspace(self.lb[i], self.ub[i], self.num) for i in range(self.lb.shape[0])])
        self.N = 10
        self.N_col = (self.num)**self.d
        self.num_operators = 24
        self.init()
      

    def init(self):
        self.tau1 = 1.0
        self.rho1 = logit(0.5)
        self.tol1 = 1e-5
        self.damping1 = 0.9
        self.s01 = 1.0
        self.INF1 = 1000000.0
        d1 = self.num_operators
        self.rho_p_full1 = np.zeros(d1)
        self.mu_p_full1 = np.zeros(d1)
        self.v_p_full1 = self.INF1 * np.ones(d1)
        self.r_full1 = self.rho_p_full1 + self.rho1

        self.tau2 = 0.5
        self.rho2 = logit(0.5)
        self.tol2 = 1e-5
        self.damping2 = 0.9
        self.s02 = 1.0
        self.INF2 = 1000000.0
        d2 = self.num_operators
        self.rho_p_full2 = np.zeros(d2)
        self.mu_p_full2 = np.zeros(d2)
        self.v_p_full2 = self.INF2 * np.ones(d2)
        self.r_full2 = self.rho_p_full2 + self.rho2

    @partial(jit, static_argnums=(0, 6))
    def loss(self, params, co, w1, w2, tau, v):
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        mu = params['mu'].sum(axis=-1)
        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)
        self.cov_f = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j, :]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
        self.cov_f_inv_K = [jnp.linalg.solve(self.K_list[i], self.cov_f[i].T).T for i in range(self.X_col.shape[0])]
        f_sample = tl.tenalg.multi_mode_dot(mu, self.cov_f_inv_K).reshape(-1)
        deri_sample = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        reg1 = (params['mu'].sum(axis=-1)**2).sum()
        ls2 = jnp.exp(params['log_ls2'])
        self.Kernel2 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list2 = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        self.cho_list2 = jnp.linalg.cholesky(self.K_list2)
        self.K_inv_list2 = jnp.linalg.inv(self.K_list2)
        mu2 = params['mu2'].sum(axis=-1)
        mu2 = tl.tenalg.multi_mode_dot(mu2, self.cho_list2)
        self.cov_f2 = [self.Kernel.get_cov(self.sensors[i], self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list2 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls2)
        self.deri_cov_times_K_inv_list2 = [[jnp.matmul(self.derivative_cov_list2[i][j], self.K_inv_list2[j, :]) for j in range(len(self.derivative_cov_list2[0]))]
                                           for i in range(len(self.derivative_cov_list2))]
        self.cov_f_inv_K2 = [jnp.linalg.solve(self.K_list2[i], self.cov_f2[i].T).T for i in range(self.X_col.shape[0])]
        f_sample2 = tl.tenalg.multi_mode_dot(mu2, self.cov_f_inv_K2).reshape(-1)
        deri_sample2 = [tl.tenalg.multi_mode_dot(mu2, self.deri_cov_times_K_inv_list2[i]) for i in range(len(self.deri_cov_times_K_inv_list2))]
        u_dx1_2 = deri_sample2[1].reshape(1, -1)
        u_2 = deri_sample2[0].reshape(1, -1)
        reg2 = (params['mu2'].sum(axis=-1)**2).sum()
        u = u.reshape(1, -1)
        u_2 = u_2.reshape(1, -1)
        library1 = jnp.concatenate((u_2, u_2**2, u_2**3, u_2**4, u, u**2, u**3, u**4, u * u_2, u**2 * u_2, u**3 * u_2, u**4 * u_2, u * u_2**2, u**2 * u_2**2, u**3 * u_2**2, u**4 * u_2**2, u * u_2**3,
                                    u**2 * u_2**3, u**3 * u_2**3, u**4 * u_2**3, u * u_2**4, u**2 * u_2**4, u**3 * u_2**4, u**4 * u_2**4),
                                   axis=0)
        library2 = jnp.concatenate((u, u_2, u**2 * u_2, u_2**2, u_2**3, u_2**4, u**2, u**3, u**4, u * u_2, u**3 * u_2, u**4 * u_2, u * u_2**2, u**2 * u_2**2, u**3 * u_2**2, u**4 * u_2**2, u * u_2**3,
                                    u**2 * u_2**3, u**3 * u_2**3, u**4 * u_2**3, u * u_2**4, u**2 * u_2**4, u**3 * u_2**4, u**4 * u_2**4),
                                   axis=0)
        KL = 0.5 * reg1 + 0.5 * reg2
        elbo = -KL - tau * jnp.sum(jnp.square(f_sample.reshape(-1) - self.Y1.reshape(-1))) - tau * jnp.sum(jnp.square(f_sample2.reshape(-1) - self.Y2.reshape(-1))) - co * v * jnp.sum(
            jnp.square(u_dx1.reshape(-1, 1) - jnp.matmul(library1.T, w1.reshape(-1, 1)))) - co * v * jnp.sum(jnp.square(u_dx1_2.reshape(-1, 1) - jnp.matmul(library2.T, w2.reshape(-1, 1))))
        return -elbo.sum()

    # @partial(jit, static_argnums=(0, ))
    def pred(self, params_f,Col,Xtest):
        ls = jnp.exp(params_f['log_ls'])
        ls2 = jnp.exp(params_f['log_ls2'])
        X_col = Col[0]
        mu = params_f['mu'].sum(axis=-1)
        mu2 = params_f['mu2'].sum(axis=-1)
        kernel = Kernel(1e-2, RBF_kernel_u_1d())
        cov_f = kernel.get_cov(Xtest, X_col, ls)
        K_list = jnp.array([kernel.get_kernel_matrix(Col[i, :], ls[i]) for i in range(Col.shape[0])])
        cho_list = np.linalg.cholesky(K_list)
        mu = tl.tenalg.multi_mode_dot(mu, cho_list)
        K_inv_list = jnp.linalg.inv(K_list)
        weights = tl.tenalg.multi_mode_dot(mu.reshape(200, ), K_inv_list)
        pred1 = jnp.matmul(cov_f, weights.reshape(-1, 1))
        kernel = Kernel(1e-2, RBF_kernel_u_1d())
        cov_f2 = kernel.get_cov(Xtest, X_col, ls2)
        K_list2 = jnp.array([kernel.get_kernel_matrix(Col[i, :], ls2[i]) for i in range(Col.shape[0])])
        cho_list2 = np.linalg.cholesky(K_list2)
        mu2 = tl.tenalg.multi_mode_dot(mu2, cho_list2)
        K_inv_list2 = jnp.linalg.inv(K_list2)
        weights2 = tl.tenalg.multi_mode_dot(mu2.reshape(200, ), K_inv_list2)
        pred2 = jnp.matmul(cov_f2, weights2.reshape(-1, 1))
        return pred1.reshape(-1), pred2.reshape(-1)

    @partial(jit, static_argnums=(0,7))
    def step(self, params, opt_state, co, w1, w2, tau, v):
        loss, d_params = jax.value_and_grad(self.loss)(params, co, w1, w2, tau, v)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def init_EP(self, params):
        mu = params['mu'].sum(axis=-1)
        mu2 = params['mu2'].sum(axis=-1)
        u_sample = mu
        ls = np.exp(params['log_ls'])
        kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        K_list = np.array([kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        cho_list = np.linalg.cholesky(K_list)
        mu = tl.tenalg.multi_mode_dot(mu, cho_list)
        self.K_inv_list = np.linalg.inv(K_list)
        self.derivative_cov_list = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j, :]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
        u_sample = mu
        deri_sample = [tl.tenalg.multi_mode_dot(u_sample, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        ls2 = np.exp(params['log_ls2'])
        self.Kernel2 = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list2 = np.array([self.Kernel.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        self.cho_list2 = np.linalg.cholesky(self.K_list2)
        mu2 = tl.tenalg.multi_mode_dot(mu2, self.cho_list2)
        self.K_inv_list2 = np.linalg.inv(self.K_list2)

        self.derivative_cov_list2 = self.Kernel.get_derivative_cov_1d(self.X_col.T, ls2)
        self.deri_cov_times_K_inv_list2 = [[jnp.matmul(self.derivative_cov_list2[i][j], self.K_inv_list2[j, :]) for j in range(len(self.derivative_cov_list2[0]))]
                                           for i in range(len(self.derivative_cov_list2))]
        u_sample2 = mu2
        deri_sample2 = [tl.tenalg.multi_mode_dot(u_sample2, self.deri_cov_times_K_inv_list2[i]) for i in range(len(self.deri_cov_times_K_inv_list2))]
        u_dx1_2 = deri_sample2[1].reshape(1, -1)
        u_2 = deri_sample2[0].reshape(1, -1)
        library1 = jnp.concatenate((u_2, u_2**2, u_2**3, u_2**4, u, u**2, u**3, u**4, u * u_2, u**2 * u_2, u**3 * u_2, u**4 * u_2, u * u_2**2, u**2 * u_2**2, u**3 * u_2**2, u**4 * u_2**2, u * u_2**3,
                                    u**2 * u_2**3, u**3 * u_2**3, u**4 * u_2**3, u * u_2**4, u**2 * u_2**4, u**3 * u_2**4, u**4 * u_2**4),
                                   axis=0)
        library2 = jnp.concatenate((u, u_2, u**2 * u_2, u_2**2, u_2**3, u_2**4, u**2, u**3, u**4, u * u_2, u**3 * u_2, u**4 * u_2, u * u_2**2, u**2 * u_2**2, u**3 * u_2**2, u**4 * u_2**2, u * u_2**3,
                                    u**2 * u_2**3, u**3 * u_2**3, u**4 * u_2**3, u * u_2**4, u**2 * u_2**4, u**3 * u_2**4, u**4 * u_2**4),
                                   axis=0)
        X1 = np.array(library1.T)
        y1 = np.array(u_dx1.reshape(-1))
        X2 = np.array(library2.T)
        y2 = np.array(u_dx1_2.reshape(-1))
        self.S1 = np.linalg.inv(np.diag(1.0 / self.v_p_full1) + self.tau1 * (X1.T @ X1))
        self.mu_full1 = self.S1 @ (self.mu_p_full1 / self.v_p_full1 + self.tau1 * (X1.T @ y1))
        self.v_full1 = np.diag(self.S1)
        self.S2 = np.linalg.inv(np.diag(1.0 / self.v_p_full2) + self.tau2 * (X2.T @ X2))
        self.mu_full2 = self.S2 @ (self.mu_p_full2 / self.v_p_full2 + self.tau2 * (X2.T @ y2))
        self.v_full2 = np.diag(self.S2)

    def spike_and_slab_EP(self, params):
        mu = params['mu'].sum(axis=-1)
        mu2 = params['mu2'].sum(axis=-1)
        u_sample = mu
        ls = np.exp(params['log_ls'])
        kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        K_list = np.array([kernel.get_kernel_matrix(self.X_col[i, :], ls[i]) for i in range(self.X_col.shape[0])])
        cho_list = np.linalg.cholesky(K_list)
        mu = tl.tenalg.multi_mode_dot(mu, cho_list)
        K_inv_list = np.linalg.inv(K_list)
        derivative_cov_list = kernel.get_derivative_cov_1d(self.X_col.T, ls)
        deri_cov_times_K_inv_list = [[np.matmul(derivative_cov_list[i][j], K_inv_list[j, :]) for j in range(len(derivative_cov_list[0]))]
                                          for i in range(len(derivative_cov_list))]
        u_sample = mu
        deri_sample = [tl.tenalg.multi_mode_dot(u_sample, deri_cov_times_K_inv_list[i]) for i in range(len(deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        ls2 = np.exp(params['log_ls2'])
        kernel2 = Kernel(self.jitter, RBF_kernel_u_1d())
        K_list2 = np.array([kernel2.get_kernel_matrix(self.X_col[i, :], ls2[i]) for i in range(self.X_col.shape[0])])
        cho_list2 = np.linalg.cholesky(K_list2)
        mu2 = tl.tenalg.multi_mode_dot(mu2, cho_list2)
        K_inv_list2 = np.linalg.inv(K_list2)

        derivative_cov_list2 = kernel2.get_derivative_cov_1d(self.X_col.T, ls2)
        deri_cov_times_K_inv_list2 = [[np.matmul(derivative_cov_list2[i][j], K_inv_list2[j, :]) for j in range(len(derivative_cov_list2[0]))]
                                           for i in range(len(derivative_cov_list2))]
        u_sample2 = mu2
        deri_sample2 = [tl.tenalg.multi_mode_dot(u_sample2, deri_cov_times_K_inv_list2[i]) for i in range(len(deri_cov_times_K_inv_list2))]
        u_dx1_2 = deri_sample2[1].reshape(1, -1)
        u_2 = deri_sample2[0].reshape(1, -1)
        library1 = np.concatenate((u_2, u_2**2, u_2**3, u_2**4, u, u**2, u**3, u**4, u * u_2, u**2 * u_2, u**3 * u_2, u**4 * u_2, u * u_2**2, u**2 * u_2**2, u**3 * u_2**2, u**4 * u_2**2, u * u_2**3,
                                    u**2 * u_2**3, u**3 * u_2**3, u**4 * u_2**3, u * u_2**4, u**2 * u_2**4, u**3 * u_2**4, u**4 * u_2**4),
                                   axis=0)
        library2 = np.concatenate((u, u_2, u**2 * u_2, u_2**2, u_2**3, u_2**4, u**2, u**3, u**4, u * u_2, u**3 * u_2, u**4 * u_2, u * u_2**2, u**2 * u_2**2, u**3 * u_2**2, u**4 * u_2**2, u * u_2**3,
                                    u**2 * u_2**3, u**3 * u_2**3, u**4 * u_2**3, u * u_2**4, u**2 * u_2**4, u**3 * u_2**4, u**4 * u_2**4),
                                   axis=0)
        X1 = np.array(library1.T)
        y1 = np.array(u_dx1.reshape(-1))
        X2 = np.array(library2.T)
        y2 = np.array(u_dx1_2.reshape(-1))
        big_inds1 = np.where(abs(self.mu_full1) > 1e-4)[0]
        big_inds2 = np.where(abs(self.mu_full2) > 1e-4)[0]
        X1 = X1[:, big_inds1]
        X2 = X2[:, big_inds2]
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
        for i in range(self.num_operators):
            if i not in big_inds1:
                self.mu_full1[i] = 0

        for i in range(self.num_operators):
            if i not in big_inds2:
                self.mu_full2[i] = 0
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

        self.sel_prob1 = expit(self.r1)
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
                if self.sel_prob1[ind1] >= 0.5:
                    new_mu1[i] = self.mu1[ind1]
                else:
                    self.mu_full1[i] = 0
                ind1 = ind1 + 1
            else:
                self.mu_full1[i] = 0
        self.sel_prob2 = expit(self.r2)
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
                if self.sel_prob2[ind2] >= 0.5:
                    new_mu2[i] = self.mu2[ind2]
                else:
                    self.mu_full2[i] = 0
                ind2 = ind2 + 1
            else:
                self.mu_full2[i] = 0
                
        print(sel_prob1,sel_prob2)
        return new_mu1.reshape(-1), new_mu2.reshape(-1)

    def train(self):

        params_f = {
            'mu': jnp.zeros((self.num, 100)),
            'log_ls': jnp.array([0.2]),
            'mu2': jnp.zeros((self.num, 100)),
            'log_ls2': jnp.array([0.2]),
        }
        self.optimizer = optax.adam(1e-3)
        opt_state = self.optimizer.init(params_f)
        co = 0.0
        w1 = jnp.zeros(self.num_operators)
        w2 = jnp.zeros(self.num_operators)
        start_EQD = 400
        EQD_interval = 100
        t1=1000
        t2=40
        for i in range(400000):
            params_f, opt_state, _ = self.step(params_f, opt_state, co, w1, w2, t1, t2)
            if i + 1 == start_EQD:
                self.init_EP(params_f)
                w1, w2 = self.spike_and_slab_EP(params_f)
                co = 1.0
                schedule = optax.exponential_decay(1e-3, 100000, 1e-3, end_value=1e-6)
                self.optimizer = optax.adam(schedule)
                opt_state = self.optimizer.init(params_f)

            if (i + 1) <= start_EQD and (i + 1) % 100 == 0:
                print(i)
            if (i + 1) % 200 == 0 and (i + 1) > start_EQD:
                print(i,"Learned ", w1, "\n",w2)
            if (i + 1) % EQD_interval == 0 and (i + 1) > start_EQD:
                w1, w2 = self.spike_and_slab_EP(params_f)
            if (i+1)==30000:
                self.tau1=10000
                self.tau2=10000
                t2=200




X_train=np.load('xtrain_osc.npy')
Y_train=np.load('ytrain_osc.npy')
X_test=np.load('xtest_osc.npy')
Y_test=np.load('ytest_osc.npy')
sensors=[X_train.reshape(-1)]

MEQD = EQD(sensors, Y_train, X_test, Y_test,X_train)
MEQD.train()
