import matplotlib.pyplot as plt
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


np.random.seed(0)
random.seed(0)
print("Jax on", xla_bridge.get_backend().platform)
tl.set_backend('jax')


class EQD:
    def __init__(self, X_train, Y_train, X_test, Y_test, sensors, sensors_test):
        self.sensors = sensors
        self.X = X_train
        self.Y = Y_train
        self.jitter = 1e-6
        self.num = 200
        self.d = X_train.shape[1]
        self.X_test = sensors_test
        self.Y_test = Y_test.reshape(-1)
        self.lb = X_test.min(axis=0)
        self.ub = X_test.max(axis=0)
        self.X_col = np.array([np.linspace(self.lb[i], self.ub[i], self.num) for i in range(self.lb.shape[0])])
        self.N = X_train.shape[0]
        self.N_col = (self.num)**self.d
        self.num_operators = 24
        self.init_spike_and_slab()
        self.bound=0.01


    def init_spike_and_slab(self):
        self.tau = 500
        self.rho = logit(0.4)   
        self.tol = 1e-8
        self.damping = 0.9
        self.s0 = 1.0
        self.INF = 100000.0
        d = self.num_operators
        self.rho_p_full = np.zeros(d)
        self.mu_p_full = np.zeros(d)
        self.v_p_full = self.INF * np.ones(d)
        self.r_full = self.rho_p_full + self.rho

    @partial(jit, static_argnums=(0, ))
    def loss(self, params, co, w, tau, v):
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        mu = params['mu'].sum(axis=-1)
        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)
        self.cov_f = [self.Kernel.get_cov(self.sensors[i], self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list = self.Kernel.get_derivative_cov(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
        self.cov_f_inv_K = [jnp.linalg.solve(self.K_list[i], self.cov_f[i].T).T for i in range(self.X_col.shape[0])]
        f_sample = tl.tenalg.multi_mode_dot(mu, self.cov_f_inv_K).reshape(-1)
        deri_sample = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        u_dx2 = deri_sample[2].reshape(1, -1)
        u_ddx1 = deri_sample[3].reshape(1, -1)
        u_dddx1 = deri_sample[4].reshape(1, -1)
        u_ddddx1 = deri_sample[5].reshape(1, -1)
        library = jnp.concatenate((u * u_dx1, u_ddx1, u_ddddx1, u_dx1, u_dddx1, u, u**2, u**3, u**4, u**2 * u_dx1, u**3 * u_dx1, u**4 * u_dx1, u * u_ddx1, u**2 * u_ddx1, u**3 * u_ddx1, u**4 * u_ddx1,
                                   u * u_dddx1, u**2 * u_dddx1, u**3 * u_dddx1, u**4 * u_dddx1, u * u_ddddx1, u**2 * u_ddddx1, u**3 * u_ddddx1, u**4 * u_ddddx1),
                                  axis=0)
        reg = (params['mu'].sum(axis=-1)**2).sum()
        KL = 0.5 * reg
        elbo = -KL - tau * jnp.sum(jnp.square(f_sample.reshape(-1) - self.Y.reshape(-1))) - co * v * jnp.sum(jnp.square(u_dx2.reshape(-1, 1) - jnp.matmul(library.T, w.reshape(-1, 1))))
        return -elbo.sum()

    @partial(jit, static_argnums=(0, ))
    def pred(self, params_f):
        ls = jnp.exp(params_f['log_ls'])
        X_col = self.X_col
        mu = params_f['mu'].sum(axis=-1)
        cov_f = [self.Kernel.get_cov(self.X_test[i], X_col[i], ls[i]) for i in range(self.d)]
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        weights = tl.tenalg.multi_mode_dot(mu, self.K_inv_list)
        pred = tl.tenalg.multi_mode_dot(weights, cov_f).reshape(-1)
        return jnp.array(pred.reshape(-1))

    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, co, w, tau, v):
        loss, d_params = jax.value_and_grad(self.loss)(params, co, w, tau, v)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def init_EP(self, params):
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        mu = params['mu'].sum(axis=-1)
        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)
        self.cov_f = [self.Kernel.get_cov(self.sensors[i], self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list = self.Kernel.get_derivative_cov(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
        self.cov_f_inv_K = [jnp.linalg.solve(self.K_list[i], self.cov_f[i].T).T for i in range(self.X_col.shape[0])]
        f_sample = tl.tenalg.multi_mode_dot(mu, self.cov_f_inv_K).reshape(-1)
        deri_sample = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        u_dx2 = deri_sample[2].reshape(1, -1)
        u_ddx1 = deri_sample[3].reshape(1, -1)
        u_dddx1 = deri_sample[4].reshape(1, -1)
        u_ddddx1 = deri_sample[5].reshape(1, -1)
        library = jnp.concatenate((u * u_dx1, u_ddx1, u_ddddx1, u_dx1, u_dddx1, u, u**2, u**3, u**4, u**2 * u_dx1, u**3 * u_dx1, u**4 * u_dx1, u * u_ddx1, u**2 * u_ddx1, u**3 * u_ddx1, u**4 * u_ddx1,
                                   u * u_dddx1, u**2 * u_dddx1, u**3 * u_dddx1, u**4 * u_dddx1, u * u_ddddx1, u**2 * u_ddddx1, u**3 * u_ddddx1, u**4 * u_ddddx1),
                                  axis=0)

        X = np.array(library.T)
        y = np.array(u_dx2.reshape(-1))
        self.S = np.linalg.inv(np.diag(1.0 / self.v_p_full) + self.tau * (X.T @ X))
        self.mu_full = self.S @ (self.mu_p_full / self.v_p_full + self.tau * (X.T @ y))
        self.v_full = np.diag(self.S)

    @partial(jit, static_argnums=(0, ))
    def get_lib(self, params):
        ls = jnp.exp(params['log_ls'])
        self.Kernel = Kernel(self.jitter, RBF_kernel_u_1d())
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        mu = params['mu'].sum(axis=-1)
        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)
        self.cov_f = [self.Kernel.get_cov(self.sensors[i], self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])]
        self.derivative_cov_list = self.Kernel.get_derivative_cov(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
        self.cov_f_inv_K = [jnp.linalg.solve(self.K_list[i], self.cov_f[i].T).T for i in range(self.X_col.shape[0])]
        f_sample = tl.tenalg.multi_mode_dot(mu, self.cov_f_inv_K).reshape(-1)
        deri_sample = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_sample[1].reshape(1, -1)
        u = deri_sample[0].reshape(1, -1)
        u_dx2 = deri_sample[2].reshape(1, -1)
        u_ddx1 = deri_sample[3].reshape(1, -1)
        u_dddx1 = deri_sample[4].reshape(1, -1)
        u_ddddx1 = deri_sample[5].reshape(1, -1)
        library = jnp.concatenate((u * u_dx1, u_ddx1, u_ddddx1, u_dx1, u_dddx1, u, u**2, u**3, u**4, u**2 * u_dx1, u**3 * u_dx1, u**4 * u_dx1, u * u_ddx1, u**2 * u_ddx1, u**3 * u_ddx1, u**4 * u_ddx1,
                                   u * u_dddx1, u**2 * u_dddx1, u**3 * u_dddx1, u**4 * u_dddx1, u * u_ddddx1, u**2 * u_ddddx1, u**3 * u_ddddx1, u**4 * u_ddddx1),
                                  axis=0)

        return library, u_dx2

    def spike_and_slab_EP(self, params):
        library, u_dx2 = self.get_lib(params)
        X = np.array(library.T)
        y = np.array(u_dx2.reshape(-1))
        big_inds = np.where(abs(self.mu_full) > self.bound)[0]
        X = X[:, big_inds]
        self.r = self.r_full[big_inds]
        self.mu = self.mu_full[big_inds]
        self.v_p = self.v_p_full[big_inds]
        self.v = self.v_full[big_inds]
        self.rho_p = self.rho_p_full[big_inds]
        self.mu_p = self.mu_p_full[big_inds]
        for i in range(self.num_operators):
            if i not in big_inds:
                self.mu_full[i] = 0
        for it in range(10000):
            old_mu = self.mu.copy()
            v_inv_not = 1 / self.v - 1 / self.v_p
            v_not = 1 / v_inv_not
            mu_not = v_not * (self.mu / self.v - self.mu_p / self.v_p)
            v_tilt = 1 / (1 / v_not + 1 / self.s0)
            mu_tilt = v_tilt * (mu_not / v_not)
            log_h = normal.logpdf(mu_not, scale=np.sqrt(v_not + self.s0))
            log_g = normal.logpdf(mu_not, scale=np.sqrt(v_not))
            rho_p = log_h - log_g
            sel_prob = expit(self.rho + rho_p)
            mu = sel_prob * mu_tilt
            v = sel_prob * (v_tilt + (1.0 - sel_prob) * mu_tilt**2)
            self.rho_p = self.damping * rho_p + (1 - self.damping) * self.rho_p
            v_p_inv = 1 / v - v_inv_not
            v_p_inv[v_p_inv <= 0] = 1 / self.INF
            v_p_inv_mu = mu / v - mu_not / v_not
            v_p_inv = self.damping * v_p_inv + (1 - self.damping) * 1 / self.v_p
            v_p_inv_mu = self.damping * v_p_inv_mu + (1 - self.damping) * self.mu_p / self.v_p
            self.v_p = 1 / v_p_inv
            self.mu_p = self.v_p * v_p_inv_mu
            self.r = self.rho_p + self.rho
            self.S = np.linalg.inv(np.diag(1.0 / self.v_p) + self.tau * (X.T @ X))
            self.mu = self.S @ (self.mu_p / self.v_p + self.tau * (X.T @ y))
            self.v = np.diag(self.S)
            diff = np.sum((old_mu - self.mu)**2)
        self.sel_prob = expit(self.r)
        self.r_full[big_inds] = self.r
        self.mu_full[big_inds] = self.mu
        self.v_p_full[big_inds] = self.v_p
        self.v_full = self.v_full.copy()
        self.v_full[big_inds] = self.v
        self.rho_p_full[big_inds] = self.rho_p
        self.mu_p_full[big_inds] = self.mu_p
        ind = 0
        new_mu = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if i in big_inds:
                if self.sel_prob[ind] >= 0.5:
                    new_mu[i] = self.mu[ind]
                else:
                    self.mu_full[i] = 0
                ind = ind + 1
            else:
                self.mu_full[i] = 0
        print(self.sel_prob)
        return new_mu.reshape(-1)

    def train(self):
        params_f = {
            'mu': np.zeros((self.num, self.num, 100)),
            'log_ls': np.array([1.0, 1.5]),
        }
        optimizer1 = optax.adam(5e-4)
        opt_state = optimizer1.init(params_f)
        co = 0.0
        optimizer = optimizer1
        w = jnp.zeros(self.num_operators)
        start_EQD = 2000
        EQD_interval = 200
        for i in range(6000000):
            params_f, opt_state, _ = self.step(optimizer, params_f, opt_state, co, w, 5000,5000)
            if i + 1 == start_EQD:
                self.init_EP(params_f)
                w = self.spike_and_slab_EP(params_f)
                co = 1.0
                optimizer2 = optax.adam(5e-4)
                opt_state = optimizer2.init(params_f)
                optimizer = optimizer2
            if i + 1 == 20000:
                self.bound=0.1
            if (i + 1) <= start_EQD and (i + 1) % 100 == 0:
                print

            if (i + 1) % 200 == 0 and (i + 1) > start_EQD:
                print("Learned ", w)
            if (i + 1) % EQD_interval == 0 and (i + 1) > start_EQD:
                w = self.spike_and_slab_EP(params_f)



Xtrain = np.load('Xtrain_kura.npy')
Ttrain = np.load('Ttrain_kura.npy')
Xtest = np.load('Xtest_kura.npy')
Ttest = np.load('Ttest_kura.npy')
X_train = np.load('xtrain_kura.npy')
Y_train = np.load('ytrain_kura.npy')
X_test = np.load('xtest_kura.npy')
Y_test = np.load('ytest_kura.npy')

sensors = [Xtrain, Ttrain]
sensors_test = [Xtest, Ttest]





MEQD = EQD(X_train, Y_train, X_test, Y_test, sensors, sensors_test)
MEQD.train()
