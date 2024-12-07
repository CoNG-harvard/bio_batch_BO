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


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

from gpytorch.constraints.constraints import Interval

import re
import os
import copy

from src.GP_models import ExactGPModel, TorchGPModel


## Set random seed
torch.manual_seed(1)

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
    seed = 0
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
    seed: integer
        Random seed.
    -------------
    Output
    -------------
    proposed new a for states s, shape = [n_agents, action_dim]
    -------------
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
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
                               bounds=(0, 1), n_restarts=10, seed=0):
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
        seed: integer
            Random seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
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



def plot_ucb_surfaces(N = 30, pt = 0.3, models = [], beta = 0.2, elev = 30, azim = 3, plot_ucb = False, 
    common_zaxis = False, same_color = False, show_fn = True,
    n_prior = None, n_new = None, plot_poly = True, Xnew = None, Ynew = None, stride = 1):


    a1 = np.linspace(0.,1.0,N)
    a2 = np.linspace(0.,1.0,N)
    A1,A2 = np.meshgrid(a1,a2)
    # ucb = np.empty((N,N))
    K = len(models)
    mu = np.empty((K,N,N))
    std = np.empty((K,N,N))
    plt.rcParams["font.family"] = "Arial"
    fig, axs = plt.subplots(1, len(models),figsize=(3.5 * (len(models)) + 5, 3.5), subplot_kw = {'projection': '3d'})

    for k in np.arange(len(models)):
        model = models[k]
        for i in np.arange(N):
            for j in np.arange(N):
                mu_ij, std_ij = model.predict(np.asarray([pt,A1[i,j], A2[i,j]]), return_std = True)
                mu[k,i,j] = mu_ij.squeeze()
                std[k,i,j] = std_ij.squeeze()


    alpha = 0.2

    cmap = plt.get_cmap("Spectral_r")

    # colors = [cmap(0), cmap(0.85)]
    colors = [cmap(0), cmap(0.8)]
    mean_labels = ['prior mean', 'posterior mean']
    ub_labels = ['prior lower/upper uncertainty bounds', 'posterior lower/upper uncertainty bounds']
    for k in np.arange(0,len(models)):
        if same_color == False:
            if plot_ucb == True:
                surf = axs[k].plot_surface(A1 , A2 , mu[k,:,:], 
                    color =  colors[k], shade = False, antialiased = False, rstride = stride, cstride = stride)
            surf_up = axs[k].plot_surface(A1, A2 , mu[k,:,:] + beta * std[k,:,:], 
                color = colors[k], alpha = alpha, shade = False, antialiased = False, rstride = stride, cstride = stride)
            surf_down = axs[k].plot_surface(A1 , A2, mu[k,:,:] - beta * std[k,:,:], 
                color=colors[k], alpha = alpha, shade = False, antialiased = False, rstride = stride, cstride = stride)


            #set x,y lims
            axs[k].set_xlim(-0.1,1.1)
            axs[k].set_ylim(-0.1, 1.1)

            #add legend
            legend_elements = [Line2D([0], [0], color=colors[k], lw=4, label=mean_labels[k]),
                   Line2D([0], [0], color=colors[k], lw=4, label=ub_labels[k])]
            legend = axs[k].legend(handles=legend_elements, loc = 'upper right', bbox_to_anchor=(1.2, 1))
            t = 0
            for legend_handle in legend.legend_handles:
                if t == 1:
                    legend_handle.set_alpha(alpha)
                t += 1

            if plot_poly == True:
                x = A1 
                y = A2
                ucb = mu[k,:,:] + beta * std[k,:,:]
                lcb = mu[k,:,:] - beta*std[k,:,:]
                verts = []
                for m in np.arange(N*N-1):
                    verts.append(np.asarray([[x.flatten()[m], y.flatten()[m], ucb.flatten()[m]],
                        [x.flatten()[m], y.flatten()[m], lcb.flatten()[m]],
                        [x.flatten()[m+1], y.flatten()[m+1], lcb.flatten()[m+1]],
                        [x.flatten()[m+1], y.flatten()[m+1], ucb.flatten()[m+1]]]))
                poly = Poly3DCollection(verts, facecolors = colors[k], alpha = 0.01)
                axs[k].add_collection3d(poly)



    y_eps_new = 0.01
    for l in range(len(Ynew)):
        # print("l=%d" %l,Xnew[l,0], Xnew[l,1], Ynew[l] )
        if l == 0: #only label first new data
            axs[1].scatter(Xnew[l,0], Xnew[l,1], Ynew[l] + y_eps_new, color = cmap(0.7), marker = "X", s = 40, label = "new observations")
        elif l == 1: #for debugging purposes
            axs[1].scatter(Xnew[l,0], Xnew[l,1], Ynew[l] + y_eps_new, color = cmap(0.7), s = 40, marker = "X")
        else:
            axs[1].scatter(Xnew[l,0], Xnew[l,1], Ynew[l] + y_eps_new, color = cmap(0.7), s= 40, marker = "X")


    legend_elements = [Line2D([0], [0], color=colors[1], lw=4, label=mean_labels[1]),
                   Line2D([0], [0], color=colors[1], lw=4, label=ub_labels[1],alpha=alpha)]
    # Retrieve automatic legend elements
    handles, labels = axs[1].get_legend_handles_labels()
    
    # Prepend manual elements to the automatic ones
    handles = legend_elements + handles  # Combine manually defined and automatic handles
    
    # Update the legend with the combined handles
    axs[1].legend(handles=handles, loc='upper right', bbox_to_anchor=(1.2, 1))


    # Add labels and title
    titles = ['(Prior) GP belief', 
    '(Posterior) GP belief']
    for i in np.arange(len(models)):
        axs[i].set_xlabel('Amplitude (normalized)')
        axs[i].set_ylabel('Frequency (normalized)')
        axs[i].set_zlabel("Change in pseudotime")
        axs[i].view_init(elev = elev, azim = azim)
        axs[i].set_title(titles[i])
        axs[i].set_zlim(-0.2,0.3)

    axs[0].set_zlabel("Change in pseudotime", labelpad=-0.2)  # Adjust padding
    axs[0].set_box_aspect(None, zoom=0.88)
    axs[1].set_box_aspect(None, zoom=0.88)
    # Adjust the width space between subplots
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.5, wspace=1.5)
    plt.show()


