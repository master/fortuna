# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cyclical Stochastic Gradient Gradient Markov Chain Monte Carlo (SG-MCMC) method

# %% [markdown]
# Stochastic Gradient MCMC algorithms can be used to sample from the posterior distribution of large models, such as Bayesian Neural Networks.
#
# However, SG-MCMC algorithms can be inefficient at exploring multimodal distributions, which are typical of those models. To see this, let's consider a simple example with a mixture of four Gaussian distributions:

# %%
import itertools

import jax
import jax.scipy as jsp
import jax.numpy as jnp

N = 10_000
lmbda = 1/25
positions = [-4, -2, 0, 2, 4]
mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
sigma = 0.03 * jnp.eye(2)

def sample(rng_key):
    choose_key, sample_key = jax.random.split(rng_key)
    samples = jax.random.multivariate_normal(sample_key, mu, sigma)
    return jax.random.choice(choose_key, samples)

rng_key = jax.random.PRNGKey(0)
samples = jax.vmap(sample)(jax.random.split(rng_key, N))

# %% [markdown]
# We generate $N = 10000$ samples from the target distribution to plot the ground truth density:

# %%
import matplotlib.pylab as plt

import numpy as np
from scipy.stats import gaussian_kde

def plot_trajectory(ax, samples, xmin=-5, ymin=-5, xmax=5, ymax=5):
    x = [sample[0] for sample in samples]
    y = [sample[1] for sample in samples]
    ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

def plot_density(ax, samples, xmin=-5, ymin=-5, xmax=5, ymax=5, nbins=300j):
    x, y = samples[:, 0], samples[:, 1]
    xx, yy = np.mgrid[xmin:xmax:nbins, ymin:ymax:nbins]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')


# %%
fig, ax = plt.subplots()
plot_density(ax, samples)
plt.show()

# %% [markdown]
# To illustrate the SG-MCMC behavior in Fortuna, we build a dummy dataset and a model log probability function, that will be used instead of the standard classification posterior.

# %%
from fortuna.data import DataLoader

labels = jax.random.bernoulli(rng_key, shape=(len(samples),)).astype("float32")
train_data_loader = DataLoader.from_array_data((samples, labels), batch_size=1)


# %%
def log_prob(x):
    return lmbda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, mu, sigma))

def batched_log_prob(params, batch, **kwargs):
    return - jnp.sum(jax.vmap(log_prob)(batch[0])), {}


# %% [markdown]
# ### Stochastic Gradient Hamiltonian Monte Carlo

# %% [markdown]
# First, we appoximate the posterior using Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) method. To this end, we collect $K = 5000$ samples after the initial burn-in phase, while thinning the chain to reduce autocorrelation:

# %%
# %%capture
from fortuna.model import ConstantModel
from fortuna.prob_model import ProbClassifier, SGHMCPosteriorApproximator
from fortuna.prob_model import FitConfig, FitOptimizer
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import polynomial_schedule

prob_model = ProbClassifier(
    model=ConstantModel(output_dim=2),
    posterior_approximator=SGHMCPosteriorApproximator(
        step_schedule=1e-4,
        burnin_length=1000,
        n_thinning=10,
        n_samples=5_000,
    ),
    output_calibrator=None,
)

prob_model.joint._batched_log_joint_prob = batched_log_prob

status = prob_model.train(
    train_data_loader=train_data_loader,
    fit_config=FitConfig(optimizer=FitOptimizer(n_epochs=20))
)


# %% [markdown]
# The plots show that SGHMC is getting stuck in a few modes, and has hard times escaping those regions, that leads to a poor approximation of the distribution:

# %%
def sgmcmc_posterior_samples(prob_model):
    n_samples = prob_model.posterior.state.size
    samples = [prob_model.posterior.state.get(i).params["model"]["params"]["constant"]
               for i in range(n_samples)]
    return jnp.array(samples)


# %%
sghmc_samples = sgmcmc_posterior_samples(prob_model)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
plot_trajectory(axes[0], sghmc_samples)
plot_density(axes[1], sghmc_samples)
fig.tight_layout()
plt.show()

# %% [markdown]
# Next, we construct a Cyclical Stochastic Gradient Langevin Dynamics approximator. To escape modes and better explore distributions, the algorithm leverages the cyclical step size schedule, where it allocates a fraction of steps in the beginning of each cycle for mode exploration with deterministic Stochastic Gradient Descent updates.
#
# In this example, we run the algorithm for multiple cycles, while reserving 25% of steps for the exploration phase:

# %%
# %%capture
from fortuna.prob_model import CyclicalSGLDPosteriorApproximator

prob_model = ProbClassifier(
    model=ConstantModel(output_dim=2),
    posterior_approximator=CyclicalSGLDPosteriorApproximator(
        n_samples=5000,
        cycle_length=2000,
        init_step_size=0.01,
        n_thinning=10,
        exploration_ratio=0.25,
    ),
    output_calibrator=None,
)

prob_model.joint._batched_log_joint_prob = batched_log_prob

status = prob_model.train(
    train_data_loader=train_data_loader,
    fit_config=FitConfig(optimizer=FitOptimizer(n_epochs=20))
)


# %% [markdown]
# The trajectory of the sampler shows much better exploration of the distribution:

# %%
cyclical_sgld_samples = sgmcmc_posterior_samples(prob_model)

fig, ax = plt.subplots(figsize=(4, 4))
plot_trajectory(ax, cyclical_sgld_samples)
plt.show()

# %%
