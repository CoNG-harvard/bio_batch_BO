#Adapted and modified from https://github.com/mahaitongdae/dbo



import anndata as an
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider

import scipy.stats as stats
import random
import joblib
from joblib import Parallel, delayed
import multiprocessing
import itertools
import time

import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel

from scipy.stats import norm
from scipy.optimize import minimize
import torch
import gpytorch
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from gpytorch.constraints.constraints import Interval

import re
import os
import copy

from src.GP_models import ExactGPModel, TorchGPModel

#bayesian opt acquisition functions

def expected_improvement(a, s, gaussian_process, evaluated_loss, greater_is_better=True, n_params=2, s_dim = 1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        s: current s
        a: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    a_to_predict = a.reshape(-1, n_params)
    sa_to_predict = np.insert(a_to_predict,0,s)
    # print("shape", sa_to_predict.shape)

    mu, sigma = gaussian_process.predict(sa_to_predict.reshape(-1,n_params +s_dim), return_std=True)
    # print("s:", s)
    # print("sa:", sa_to_predict)
    # print("mu", mu)
    # print("sigma", sigma)
    # print("hi here", mu)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
        # print("loss_optimum", loss_optimum)
        # print("evaluated loss", evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    # print("expected improvement", expected_improvement.reshape(1,))
    return -1 * expected_improvement.reshape(1,)



def TS(a, s, gaussian_process, n_params=2, s_dim=1):
    """Thompson sampling variant of acquisition function
    For each (s,a) pair, sample f(s,a) from N(mu(s,a), sigma(s,a)^2)
    where the mu(s,a) and sigma(s,a) are based on current GP
    Arguments
    --------------
    s: curr state
    a: any input
    gaussian_process: GaussianProcessRegressor object.
        Gaussian process trained on previously evaluated hyperparameters.
    n_params: dimension of a
    -------------
    Output
    -------------
    -f(s,a) ; negative sign is because we later want to minimize -f(s,a).
    -------------
    """
    a_to_sample = a.reshape(-1, n_params)
    sa_to_sample = np.insert(a_to_sample, 0, s)
    # print("sa_to sample shaoe", sa_to_sample.shape)
    # crucial to not fix random_state, otherwise diff agents will use same TS, i.e. not adding info
    res = gaussian_process.sample_y(
        sa_to_sample.reshape(-1, s_dim + n_params), random_state=None
    ).reshape(
        1,
    )
    # print("sa", sa_to_sample)
    # print("gp sample", -res)
    if res <= 0:  # gp might return negative value
        return 0
    else:
        return -res


def ucb(a, s, gaussian_process, n_params=2, beta=0.15, s_dim=1):
    """UCB variant of acquisition function
    Arguments
    --------------
    s: single state
    a: any input
    gaussian_process: GaussianProcessRegressor object.
        Gaussian process trained on previously evaluated hyperparameters.
    n_params: dimension of action a
    -------------
    Output
    -------------
    -(mu(s,a) + beta * sigma(s,a)) ; negative sign is because we want to minimize a cost
    -------------
    """
    a_to_sample = a.reshape(-1, n_params)
    sa_to_sample = np.insert(a_to_sample, 0, s)
    mu, sigma = gaussian_process.predict(
        sa_to_sample.reshape(-1, n_params + s_dim), return_std=True
    )
    mu = np.squeeze(mu)
    # print("mu", mu)
    # print("beta * sigma", beta * sigma)
    # print("this is mu shape", mu.shape)
    # print("this is sigma shape", sigma.shape)
    acq = mu + beta * sigma
    return -1.0 * acq


def ES(
    sa,
    gp,
    state_dim=1,
    n_params=2,
    beta=0.2,
    alpha=0.0,
    a_circ=0.15,
    tr_iters=200,
    domain=[0, 1],
    init_points=5,
    noise=1e-4,
):
    """entropy search variant of action proposal
    Arguments
    --------------
    sa: (state,action) pairs, shape = [n_agents, state_dim + action_dim],
        a[i] is estimated ucb action for state s[i]
    gp: GaussianProcessRegressor object.
        Gaussian process trained on previously evaluated hyperparameters.
    n_params: dimension of action a
    beta: ucb exploration bonus
    alpha: (our belief of) observation noise variance, default 0 b/c gp should already include the noise
    -------------
    Output
    -------------
    proposed new a for states s, shape = [n_agents, action_dim]
    -------------
    """
    s = sa[:, :state_dim]
    a = sa[:, state_dim:]
    # print("these are the s",s)
    # print("these are ucb/ei a:", a)
    n_agents = sa.shape[0]
    a_dim = n_params
    best_loss = 1e5
    best_a = None
    for n_init in range(init_points):
        a_new = np.random.normal(loc=a, scale=a_circ, size=(n_agents, a_dim))
        a_new = np.minimum(1.,np.maximum(a_new,0))
        # print("initial a", a_new)
        a_new = torch.tensor(a_new, requires_grad=True, dtype=torch.float32)
        # print("initial a_new", a_new)
        s_torch = torch.tensor(s, requires_grad=False, dtype=torch.float32)
        optimizer = torch.optim.Adam([a_new], lr=0.01)
        training_iter = tr_iters
        for i in range(training_iter):
            optimizer.zero_grad()
            xucb_torch = torch.tensor(sa, dtype=torch.float32)
            xnew_torch = torch.hstack((s_torch, a_new))
            #   print("xnew torch require grad", xnew_torch.requires_grad)
            # print("xucb_torch", xucb_torch)
            # print("xnew_torch", xnew_torch)
            joint_x = torch.vstack((xucb_torch, xnew_torch))
            #   print("joint x requires grad", joint_x.requires_grad)
            # joint_x = joint_x.detach().numpy()
            _, cov = gp.predict(joint_x, return_cov=True, return_tensor = True)
            # cov = torch.tensor(cov, requires_grad=True).float()
            cov_xucb_x = cov[:n_agents, n_agents:]
            cov_xx = cov[n_agents:, n_agents:]
            # print("cov_xucb_x", cov_xucb_x)
            # print("cov_xx", cov_xx)
            loss = -torch.trace(
                torch.matmul(
                    torch.matmul(
                        cov_xucb_x,
                        torch.linalg.inv(cov_xx + noise * torch.eye(n_agents)),
                    ),
                    cov_xucb_x.T,
                )
            )
            # print("loss", loss)
            #   print("loss requires grad", loss.requires_grad)
            loss.backward()
            optimizer.step()
            # print("a new", a_new)
            # project a_new back to domain
            a_new = torch.where(
                a_new > torch.tensor(domain[1]), torch.tensor(domain[1]), a_new
            )
            a_new = torch.where(
                a_new < torch.tensor(domain[0]), torch.tensor(domain[0]), a_new
            )
            a_new.detach_()  # what's the point of this?
        if loss.clone().detach().numpy() < best_loss:
            # print("i am here, best loss", best_loss)
            a_new = a_new.clone().detach().numpy()
            best_a = np.copy(a_new)
            best_loss = loss.clone().detach().numpy()

    # rearrange a_new such that each a_new[i] gets assigned to the a[j] that is closest to it
    print("best a", best_a)
    a_new_reorg = np.copy(best_a)
    for i in range(n_agents):
        dist_i = np.linalg.norm(a[i, :] - best_a, axis=1)
        idx_i = np.argmin(dist_i)
        a_new_reorg[i] = best_a[idx_i, :]
        best_a = np.delete(
            best_a, (idx_i), axis=0
        )  # remove corresponding row from best_a
    # print("these are proposed a after reorg", a_new_reorg)
    return a_new_reorg


#BO sampling and optimization functions


def sample_next_hyperparameter(s,gaussian_process, evaluated_loss, acquisition_func = None, acq = "ucb",greater_is_better=True,
                               bounds=(0, 1), n_restarts=10):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter(s) to sample the loss function for.
    Arguments:
    ----------
        s: current state(s), numpy array
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_a = None

    best_acquisition_value = 1
    n_params = bounds.shape[0] -1 #-1 is because s is not part of the params for acquisition

    for starting_point in np.random.uniform(bounds[1:, 0], bounds[1:, 1], size=(n_restarts, n_params)):
        # print("s", s)
        # print("hi", starting_point, acq)
        # print("sampling for state:", s)
        # print("initial a for sample next fn", starting_point)
        # print("before optimizing")
        if acq == "EI":
            res = minimize(fun=acquisition_func,
                        x0=starting_point.reshape(n_params,),
                        bounds=((0,1),(0,1)),
                        method='L-BFGS-B',
                        args=(s,gaussian_process, evaluated_loss, greater_is_better, n_params))
        elif acq == "TS" or acq == "ucb":
            res = minimize(fun=acquisition_func,
                           x0 = starting_point.reshape(n_params,),
                           bounds = ((0,1), (0,1)),
                           method = 'L-BFGS-B',
                           args = (s,gaussian_process,n_params)
                           )
        # print("after optimizing")
        # print("res final func value", res.fun)
        # print("number of iters", res.nit)

        if res.fun < best_acquisition_value:
            # print("best acq value", best_acquisition_value)
            best_acquisition_value = res.fun
            best_a = res.x
    # print("a after optimizing", best_a)
    return best_a


