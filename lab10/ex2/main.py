import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

# generare de date aleatorii
np.random.seed(42)
x_new = np.sort(np.random.uniform(-5, 5, 500))
y_new = 3 * x_new**2 - 2 * x_new + 1 + np.random.normal(0, 10, 500)

dummy_data_new = np.column_stack((x_new, y_new))

x_1 = dummy_data_new[:, 0]
y_1 = dummy_data_new[:, 1]

order = 5

x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_p:
    # sd_beta = 10
    # sd_beta = np.array([10, 0.1, 0.1, 0.1, 0.1])
    sd_beta = 100
    beta = pm.Normal('beta', mu=0, sd=sd_beta, shape=order)

    mu = pm.math.dot(x_1s.T, beta)

    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
    obs = pm.Normal('obs', mu=mu, sd=sigma, observed=y_1s)

# inferenta cu model_p
with model_p:
    trace = pm.sample(2000, tune=1000, cores=1)

az.plot_posterior(trace, var_names=['beta', 'sigma'])
plt.show()

x_range = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
x_range_p = np.vstack([x_range ** i for i in range(1, order + 1)])
y_pred = np.dot(trace['beta'].mean(axis=0), x_range_p)
plt.scatter(x_1s[0], y_1s, label='Data')
plt.plot(x_range, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
