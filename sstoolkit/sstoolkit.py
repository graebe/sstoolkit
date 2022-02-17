# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:48:44 2018

TODO:
    - Check Discretization (compare discrete simulation variants)
    - Override Prediction Function in Kalman Filter with nonlinear xp from ss

@author: graebe
"""

import control
from sympy import Matrix, eye, symbols
from sympy import lambdify as lf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import GPyOpt

# Filter
from filterpy.kalman import ExtendedKalmanFilter as EKF


def concatenate_op(op1, op2):
    return op1 + op2


# Substitute Variables with Values
def subs_matr(matr, subs_list):
    assert len(subs_list[0]) == len(subs_list[1])
    matr_ = []
    for row in matr:
        row_ = []
        for el in row:
            el_ = el.subs(subs_list)
            row_.append(el_)
        matr_.append(row_)
    return matr_


class state_space():
    '''
    Nonlinear State Space Function
        xp = f(x,u)
        y  = g(x,u)
    Built with symbolic toolbox sympy.
    '''

    def __init__(self, x, u, g, xp, exogen_u=None, dt=None, linearize=False, discretize=False, lambdify=False):
        # State Space
        self.x = x
        self.u = u
        self.g = g
        self.xp = xp
        self.exogen_u = exogen_u
        # Time Constant
        self.dt = dt
        # Linearization
        if linearize:
            self.A = None
            self.B = None
            self.C = None
            self.D = None
            self.linearize_ss()
        if discretize:
            self.A_d = None
            self.B_d = None
            self.C_d = None
            self.D_d = None
            self.discretize_ss()
        if lambdify:
            self.A_fun = None
            self.B_fun = None
            self.C_fun = None
            self.D_fun = None
            self.fun_symbols = None
            self.lambdify_ss()
        # Save Setup
        self.setup = {'linearize': linearize, 'lambdify': lambdify}

    def linearize_ss(self):
        '''
        Symbolic linearization of the nonlinear state space equations by com-
        putation of jacobians:
            1. A = dxp/dx
            2. B = dxp/du
            3. C = dg /dx
            4. D = dg /du
        Writes the symbolic jacobians to self.A, self.B, self.C, self.D.
        '''
        self.A = self.xp.jacobian(self.x)
        self.B = self.xp.jacobian(self.u)
        self.C = self.g.jacobian(self.x)
        self.D = self.g.jacobian(self.u)

    def discretize_ss(self, dt=None):
        '''
        Discretizes the state space with simple euler.

        # Arguments
            dt:     Time constant for discretization. If none is provided a
                    symbolic variable dt is introduced and used.

        # Writes to object
            self.A_d:   Discrete SS Matrix A
            self.B_d:   Discrete SS Matrix B
            self.C_d:   Discrete SS Matrix C
            self.D_d:   Discrete SS Matrix D
        '''
        if dt is None:
            dt_symbol = symbols('dt')
        else:
            dt_symbol = dt
        self.A_d = eye(self.A.shape[0]) + self.A * dt_symbol
        self.B_d = self.B * dt_symbol
        self.C_d = self.C
        self.D_d = self.D

    def lambdify_ss(self, cont_ss=True, discrete_ss=True):
        '''
        Lambdification of state space equations and matrices in operation point.
        Inputs for the function are sorted as follows:
            1. States x
            2. Inputs u
            3. Exogeneous Inputs exogen_u

        # Arguments
            cont_ss:        Option if self.xp (continuous nonlin. ss) is
                            discretized
            disctrete_ss:   Option if the linear SS (A,B,C,D) is discretized

        # Writes to object:
            self.xp_fun:    Lambdified nonlinear state space
            self.g_fun:     Lambdified nonlinear measurement function
            self.A_d_fun:   Lambdified discretized SS Matrix A_d
            self.B_d_fun:   Lambdified discretized SS Matrix B
            self.C_d_fun:   Lambdified discretized SS Matrix C
            self.D_d_fun:   Lambdified discretized SS Matrix D
        Write python functions self.A_fun, self.B_fun, self.C_fun, self.D_fun
        and self.xp_fun.
        '''
        # collect symbols that serve as arguments
        fun_symbols = []
        for symb_ in self.x:
            fun_symbols.append(symb_)
        for symb_ in self.u:
            fun_symbols.append(symb_)
        if self.exogen_u is not (None):
            for symb_ in self.exogen_u:
                fun_symbols.append(symb_)
        # assert that number of symbols fits
        if len(fun_symbols) != len(set.union(self.xp.free_symbols, self.g.free_symbols)):
            print('Number of symbols collected from self.x, self.u, self.exogen_u doesnt match')
            print('number of free symbols in the state space.')
        assert len(fun_symbols) == len(set.union(self.xp.free_symbols, self.g.free_symbols))
        # create functions
        self.fun_symbols = fun_symbols
        if cont_ss:
            self.A_fun = lf(fun_symbols, self.A)
            self.B_fun = lf(fun_symbols, self.B)
            self.C_fun = lf(fun_symbols, self.C)
            self.D_fun = lf(fun_symbols, self.D)
        if discrete_ss:
            dt_symbol = symbols('dt')
            A_d_ = self.A_d.subs([(dt_symbol, self.dt)])
            B_d_ = self.B_d.subs([(dt_symbol, self.dt)])
            C_d_ = self.C_d.subs([(dt_symbol, self.dt)])
            D_d_ = self.D_d.subs([(dt_symbol, self.dt)])
            self.A_d_fun = lf(fun_symbols, A_d_)
            self.B_d_fun = lf(fun_symbols, B_d_)
            self.C_d_fun = lf(fun_symbols, C_d_)
            self.D_d_fun = lf(fun_symbols, D_d_)
        self.xp_fun = lf(fun_symbols, self.xp)
        self.g_fun = lf(fun_symbols, self.g)

    def get_linear_ss_in_op(self, op_x, op_u=None, op_exogen_u=None, discrete=False):
        '''
        Get function for linear state space equations in specific operation
        point op. Returns linearized numerical matrices A, B, C, D.
        '''
        op = []
        for op_x_ in op_x:
            op.append(op_x_)
        for op_u_ in op_u:
            op.append(op_u_)
        if op_exogen_u is not(None):
            for op_exogen_u_ in op_exogen_u:
                op.append(op_exogen_u_)
        if discrete:
            A_ = self.A_d_fun(*op)
            B_ = self.B_d_fun(*op)
            C_ = self.C_d_fun(*op)
            D_ = self.D_d_fun(*op)
        else:
            A_ = self.A_fun(*op)
            B_ = self.B_fun(*op)
            C_ = self.C_fun(*op)
            D_ = self.D_fun(*op)
        return A_, B_, C_, D_

    def simulate_step(self, x, u, dt, exogen_u=None, use_discrete_linear_ss=False):
        '''
        Simultes one time step of the nonlinear state space equations.

        # Arguments
            x:      System State
                    shape=(n_x,1)
            u:      Control Input
                    shape=(n_u,1)
            dt:     Sample Time [s]
                    shape=(1,)

        # Outputs
            x_tp1:  System State in Next Time Step
                    shape=(n_x,1)
        '''
        if (self.exogen_u is not (None)) and (exogen_u is None):
            print('SS has exogenious inputs, but no exogenious input exogen_u given.')
            assert False
        # State and Inputs as numpy array
        xp_args = []
        for x_ in np.squeeze(x, axis=1):
            xp_args.append(x_)
        for u_ in np.squeeze(u, axis=1):
            xp_args.append(u_)
        if exogen_u is not (None):
            for exogen_u_ in exogen_u:
                xp_args.append(exogen_u_)
        # Simulation
        if use_discrete_linear_ss:
            # print('Not implemented yet.')
            # x_p__ = np.dot(self.A_fun(*xp_args),x) + np.dot(self.B_fun(*xp_args),u)
            # x_tp1 = x + dt*x_p__
            # x_tp1 = np.dot(np.diag(np.ones(shape=[self.n_x]))+dt*self.A_fun(*xp_args),x) + np.dot(dt*self.B_fun(*xp_args),u)
            x_tp1 = np.dot(self.A_d_fun(*xp_args), x) + np.dot(self.B_d_fun(*xp_args), u)
        else:
            xp_ = self.xp_fun(*xp_args)
            x_tp1 = x + dt * xp_
        # Return
        return x_tp1

    def check_observability_in_op(self, op, op_exogen_u=None):
        op_ = op
        if (self.exogen_u is not (None)) and (op_exogen_u is None):
            print('SS has exogenious inputs, but no exogenious input exogen_u given.')
        if op_exogen_u is not (None):
            op_ = op + op_exogen_u
        A = self.A.subs(op_)
        C = self.C.subs(op_)
        obs_matr = control.obsv(A, C)
        observable = np.linalg.matrix_rank(np.array(obs_matr, dtype=np.float)) == A.shape[0]
        if observable:
            print('System is observable in operation point.')
        else:
            print('System is not observable in operation point.')

    @property
    def n_u(self):
        n_u = len(self.u) if self.u is not (None) else None
        return n_u

    @property
    def n_y(self):
        n_y = len(self.g) if self.g is not (None) else None
        return n_y

    @property
    def n_x(self):
        n_x = len(self.xp) if self.xp is not (None) else None
        return n_x

    @property
    def n_exogen_x(self):
        n_exogen_x = len(self.exogen_u) if self.exogen_u is not (None) else None
        return n_exogen_x


class SSTools():

    def __init__(self):
        return

    def augment_ss_by_adaptive_var(self, ss, adaptive_var, linearize=False, lambdify=False):
        x = ss.x
        xp = ss.xp
        # Append Constant Dynamics for Adaptive Parameters
        xpa = xp
        for xai in adaptive_var:
            xpa = xpa.col_join(Matrix([[0]]))
        # Concatenate State Space and Adaptive Parameters
        xa = x
        for xai in adaptive_var:
            xa = xa.col_join(Matrix([[xai]]))
        # Return
        return state_space(xa, ss.u, ss.g, xpa, exogen_u=ss.exogen_u, linearize=linearize, lambdify=lambdify, dt=ss.dt)

    def insert_parameters_in_SS(self, ss, param, linearize=False, lambdify=False):
        xpr = ss.xp.subs(param)
        gr = ss.g.subs(param)
        return state_space(ss.x, ss.u, gr, xpr, exogen_u=ss.exogen_u, linearize=linearize, lambdify=lambdify, dt=ss.dt)


class SSParamIdent():

    def __init__(self, ss, fixed_param, ident_param, ident_param_scaling, bounds, max_iter, max_err, u_meas, y_meas,
                 x_meas, exogen_u_meas):

        self.ss = ss
        self.fixed_param = fixed_param
        self.ident_param = ident_param
        self.ident_param_scaling = ident_param_scaling
        self.bounds = bounds
        self.max_iter = max_iter
        self.max_err = max_err

        self.u_meas = u_meas
        self.y_meas = y_meas
        self.exogen_u_meas = exogen_u_meas
        self.x_meas = x_meas

        self.scaley = np.expand_dims(np.std(y_meas, axis=1), axis=1)

        self.optim = []

        self.SSTools = SSTools()

    def _f_opt(self, x, verbose=True):
        # Build Parameters for Optimization
        ident_param = []
        for p, p_val, p_val_scale in zip(self.ident_param, x[0], self.ident_param_scaling):
            ident_param.append((p, p_val * p_val_scale))
        # Collect Parameters
        parameters = self.fixed_param + ident_param
        # Iterate Epoch
        self.epoch = self.epoch + 1
        if verbose:
            print('Epoch: ' + str(self.epoch))
            print(parameters)
        # Build State Space and Simulator
        ss_opt = self.SSTools.insert_parameters_in_SS(self.ss, parameters)
        ss_opt.linearize_ss()
        ss_opt.discretize_ss()
        ss_opt.lambdify_ss()
        sim = SSSimulator(ss=ss_opt,
                          dt=0.01,
                          u_meas=self.u_meas,
                          y_meas=self.y_meas,
                          x_meas=self.x_meas,
                          exogen_u_meas=self.exogen_u_meas,
                          x0=[[0], [0]])
        # Simulate
        sim.simulate()
        y_hat = sim.y_hat
        y_meas = self.y_meas
        # Calculate Error and Return
        err = (y_hat - y_meas) / self.scaley
        mse = np.mean(err ** 2)
        if verbose:
            print('Error with current parametrization: ' + str(mse))
        if mse > self.max_err:
            mse = self.max_err
            if verbose:
                print('Corrected error to ' + str(self.max_err) + '.')
        if np.isnan(mse):
            mse = self.max_err
            if verbose:
                print('Corrected error from nan to ' + str(self.max_err) + '.')
        if verbose:
            print(' ')
        return mse

    def run_parameter_ident(self):
        self.epoch = 0
        self.optim = GPyOpt.methods.BayesianOptimization(self._f_opt, self.bounds)
        self.optim.run_optimization(self.max_iter)
        p_opt = []
        for p, p_val, p_val_scale in zip(self.ident_param, self.optim.x_opt, self.ident_param_scaling):
            p_opt.append((p, p_val * p_val_scale))
        self.p_opt = p_opt
        return p_opt


class EKF_pred(EKF):
    '''
    Builds on filterpy 1.4.5
    '''

    def __init__(self, xp_fun=None, dt=None, **kwargs):
        self.xp_fun = xp_fun
        self.dt = dt
        super().__init__(**kwargs)
        return

    def predict_x(self, u=0, args=()):
        """
        Predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you.
        """
        # self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        xp_fun_args = []
        for el in np.squeeze(self.x, axis=1):
            xp_fun_args.append(el)
        for el in np.squeeze(u, axis=1):
            xp_fun_args.append(el)
        for el in np.squeeze(args, axis=1):
            xp_fun_args.append(el)
        self.x = self.x + self.xp_fun(*xp_fun_args) * self.dt

    def predict(self, u=0, args=()):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        self.predict_x(u, args)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)


class FilterWrapper():

    def __init__(self, ss, x0):
        self.ss = ss
        self.x0 = x0

    def filter_data(self, u_meas, y_meas, exogen_u_meas=None):
        '''
        Applying filter to given data set

        # Arguments
            u_meas:         Numpy array with system inputs u, shape=(n_u,T)
            y_meas:         Numpy array with system outputs y, shape=(n_y,T)
            exogen_u_meas:  Optional numpy array with exogeneous inputs for the
                            state space.

        # Outputs
            x_hat:          Estimation x_hat, shape=(n_u,T)
            y_hat:          Estimation y_hat, shape=(n_y,T)
        '''
        # check data
        T_u, T_y, T_exogen_u = u_meas.shape[1], y_meas.shape[1], exogen_u_meas.shape[1]
        assert T_u == T_y and T_u == T_exogen_u
        # Loop over Data and update predict
        x_hat = np.zeros(shape=(self.ss.n_x, T_y))
        y_hat = np.zeros(shape=(self.ss.n_y, T_y))
        for t in tqdm(range(T_y)):
            # Current Input and Measurement
            u_t = u_meas[:, [t]]
            y_t = y_meas[:, [t]]
            if exogen_u_meas is not (None):
                exogen_u_t = exogen_u_meas[:, [t]]
            else:
                exogen_u_t = None
            # Update
            self.update(y_t, args=(u_t, exogen_u_t), hx_args=(u_t, exogen_u_t))
            x_hat[:, [t]] = self.fil.x
            y_hat[:, [t]] = self.fil.y
            # Predict
            self.predict(u_t, exogen_u_t)
        # Return
        return x_hat, y_hat

    def predict_data(self, u_meas, exogen_u_meas=None):
        '''
        Predict a time series with a given data set

        # Arguments
            u_meas:         Numpy array with system inputs u, shape=(n_u,T)
            exogen_u_meas:  Optional numpy array with exogeneous inputs for the
                            state space.

        # Outputs
            x_hat:          Estimation x_hat, shape=(n_u,T)
            y_hat:          Estimation y_hat, shape=(n_y,T)
        '''
        # check data
        T_u = u_meas.shape[1]
        if exogen_u_meas is not(None):
            T_exogen_u = exogen_u_meas.shape[1]
            assert T_u == T_exogen_u
        # Loop over Data and update predict
        x_hat = np.zeros(shape=(self.ss.n_x, T_u))
        y_hat = np.zeros(shape=(self.ss.n_y, T_u))
        for t in tqdm(range(T_u)):
            # Current Input and Measurement
            u_t = u_meas[:, [t]]
            if exogen_u_meas is not (None):
                exogen_u_t = exogen_u_meas[:, [t]]
            else:
                exogen_u_t = None
            # Append
            x_hat[:, [t]] = self.fil.x
            y_hat[:, [t]] = self.fil.y
            # Predict
            self.predict(u_t, exogen_u_t)
        # Return
        return x_hat, y_hat


class EKFWrapper(FilterWrapper):

    def __init__(self, ss, x0):
        # Initialize Super Class
        super().__init__(ss, x0)
        # Set up EKF with filterpy
        self.fil = EKF_pred(xp_fun=ss.xp_fun, dt=ss.dt, dim_x=ss.n_x, dim_z=ss.n_y, dim_u=ss.n_u)
        # self.fil = EKF(dim_x=ss.n_x, dim_z=ss.n_y, dim_u=ss.n_u)
        self.fil.x = np.array(x0)

    def _update_linearized_SS(self, x_t, u_t=None, exogen_u_t=None):
        # Get Estimated State x from Filter
        x_t = self.fil.x
        # Set Dynamic Matrix
        A_d, B_d, _, _ = self.ss.get_linear_ss_in_op(op_x=x_t, op_u=u_t, op_exogen_u=exogen_u_t, discrete=True)
        self.fil.F = np.array(A_d, dtype=np.float)
        self.fil.B = np.array(B_d, dtype=np.float)

    def _gx(self, x, u, exogen_u):
        '''
        Function which takes as input the state variable (self.x) along
        with the optional arguments in hx_args, and returns the measurement
        that would correspond to that state.

        # Arguments
            x:          Current State x, shape=(n_x,1)
            u           Current Input u, shape=(n_u,1)
            exogen_u:   Current exogeneous Input exogen_u, shape=(n_eu,1)
        '''
        g_args = [*x, *u, *exogen_u]
        g_args = [arg[0] for arg in g_args]
        return self.ss.g_fun(*g_args)

    def _gJacobian_at(self, x, u, exogen_u):
        '''
        function which computes the Jacobian of the measurement function. Takes
        state variable (self.x) as input, along with the optional arguments in
        args, and returns C.

        # Arguments
            x:          Current State x, shape=(n_x,1)
            u           Current Input u, shape=(n_u,1)
            exogen_u:   Current exogeneous Input exogen_u, shape=(n_eu,1)
        '''
        C_args = [*x, *u, *exogen_u]
        C_args = [arg[0] for arg in C_args]
        return self.ss.C_fun(*C_args)

    def update(self, y_t, args=(), hx_args=()):
        self.fil.update(y_t, self._gJacobian_at, self._gx, args=args, hx_args=hx_args)
        return

    def predict(self, u_t, exogen_u_t=None):
        x_t = self.fil.x  # CHECK LINE posterior prior?
        self._update_linearized_SS(x_t, u_t, exogen_u_t)
        self.fil.predict(u=u_t, args=exogen_u_t)


class SSSimulator():

    def __init__(self, ss, dt, y_meas, u_meas, x_meas=None, exogen_u_meas=None, x0=None):

        # Stat Space Object
        self.ss = ss
        # Sample Time
        self.dt = dt
        # Data Set
        self.x_meas = x_meas
        self.u_meas = u_meas
        self.y_meas = y_meas
        self.exogen_u_meas = exogen_u_meas
        if exogen_u_meas is not None:
            T_u, T_y, T_exogen_u = u_meas.shape[1], y_meas.shape[1], exogen_u_meas.shape[1]
            assert T_u == T_y and T_u == T_exogen_u
        else:
            T_u, T_y = u_meas.shape[1], y_meas.shape[1]
            assert T_u == T_y and T_u
        self.T = T_u
        # Initial Values
        self.x0 = np.array(x0, dtype=np.float)
        # Place Holder for Reults
        self.x_hat = None
        self.y_hat = None

    def simulate(self, use_discrete_linear_ss=False):
        # Initial State
        x_hat_t = self.x0
        if self.exogen_u_meas is not None:
            y_hat_t = self.ss.g_fun(*x_hat_t[:, 0], *self.u_meas[:, 0], *self.exogen_u_meas[:, 0])
        else:
            y_hat_t = self.ss.g_fun(*x_hat_t[:, 0], *self.u_meas[:, 0])
        # Initialize Estimation
        x_hat = np.zeros(shape=(self.ss.n_x, self.y_meas.shape[1]))
        y_hat = np.zeros_like(self.y_meas)
        x_hat[:, [0]] = x_hat_t
        y_hat[:, [0]] = y_hat_t
        # Loop over Data for Simulation
        for k in tqdm(range(1, self.T)):
            # Mapping
            u_t = self.u_meas[:, k]
            if self.exogen_u_meas is not None:
                exogen_u_t = self.exogen_u_meas[:, k]
            else:
                exogen_u_t = None
            # Simulation Step
            x_hat_tp1 = self.ss.simulate_step(x=x_hat_t, u=np.expand_dims(u_t, axis=1), exogen_u=exogen_u_t, dt=self.dt,
                                              use_discrete_linear_ss=use_discrete_linear_ss)
            if self.exogen_u_meas is not (None):
                y_hat_tp1 = self.ss.g_fun(*x_hat_tp1[:,0], *u_t, *self.exogen_u_meas[:, k])  # index is k, not zero
            else:
                y_hat_tp1 = self.ss.g_fun(*x_hat_tp1[:,0], *u_t)  # index is k, not zero  # use x_hat_t instead of x_hat_tp1
            # Concatenate Results
            x_hat[:, [k]] = x_hat_tp1
            y_hat[:, [k]] = y_hat_tp1
            # Time Step
            x_hat_t = x_hat_tp1
            y_hat_t = y_hat_tp1
        # Write Results
        self.x_hat = x_hat
        self.y_hat = y_hat

    def plot_data_y(self):
        t = np.arange(0, self.T) * self.dt
        fig, ax_list = plt.subplots(self.ss.n_y, 1)
        if type(ax_list) != np.ndarray:
            ax_list = [ax_list]
        for idx, ax in enumerate(ax_list):
            ax.plot(t, self.y_meas[idx, :], label='y_meas')
            ax.plot(t, self.y_hat[idx, :], label='y_hat')
            ax.grid()
            ax.legend(loc='upper right')

    def plot_data_x(self):
        t = np.arange(0, self.T) * self.dt
        fig, ax_list = plt.subplots(3, 1)
        if type(ax_list) != np.ndarray:
            ax_list = [ax_list]
        for idx, ax in enumerate(ax_list):
            ax.plot(t, self.x_meas[idx, :], label='x_meas')
            ax.plot(t, self.x_hat[idx, :], label='x_hat')
            ax.grid()
            ax.legend(loc='upper right')




