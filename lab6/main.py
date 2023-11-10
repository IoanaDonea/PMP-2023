import pymc3 as pm
import arviz as az
import numpy as np

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]
n_prior_lambda = 10

with pm.Model() as model:
    n = pm.Poisson('n', mu=n_prior_lambda)

    for theta in theta_values:
        Y_obs = pm.Binomial('Y_obs', n=n, p=theta, observed=Y_values)

    trace = pm.sample(2000, tune=1000, cores=1)

az.plot_posterior(trace, var_names=['n'], round_to=2, credible_interval=0.95, point_estimate='mean', rope=[9.5, 10.5])
