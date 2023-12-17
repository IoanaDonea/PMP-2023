import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

np.random.seed(42)
x_new = np.sort(np.random.uniform(-5, 5, 500))
y_new = 3 * x_new**2 - 2 * x_new + 1 + np.random.normal(0, 10, 500)

dummy_data_new = np.column_stack((x_new, y_new))

x_1 = dummy_data_new[:, 0]
y_1 = dummy_data_new[:, 1]

order_cubic = 3

# generare date pentru modelul cubic
x_1p_cubic = np.vstack([x_1 ** i for i in range(1, order_cubic + 1)])
x_1s_cubic = (x_1p_cubic - x_1p_cubic.mean(axis=1, keepdims=True)) / x_1p_cubic.std(axis=1, keepdims=True)
y_1s_cubic = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_cubic:
    sd_beta_cubic = 100
    beta_cubic = pm.Normal('beta_cubic', mu=0, sd=sd_beta_cubic, shape=order_cubic)

    mu_cubic = pm.math.dot(x_1s_cubic.T, beta_cubic)

    sigma_cubic = pm.HalfCauchy('sigma_cubic', beta=10, testval=1.0)
    obs_cubic = pm.Normal('obs_cubic', mu=mu_cubic, sd=sigma_cubic, observed=y_1s_cubic)

    trace_cubic = pm.sample(2000, tune=1000, cores=1)

# model liniar
with pm.Model() as model_linear:
    beta_linear = pm.Normal('beta_linear', mu=0, sd=100, shape=2)
    mu_linear = beta_linear[0] + beta_linear[1] * x_1s_cubic[0]
    sigma_linear = pm.HalfCauchy('sigma_linear', beta=10, testval=1.0)
    obs_linear = pm.Normal('obs_linear', mu=mu_linear, sd=sigma_linear, observed=y_1s_cubic)

    trace_linear = pm.sample(2000, tune=1000, cores=1)

# model patratic
with pm.Model() as model_quadratic:
    beta_quadratic = pm.Normal('beta_quadratic', mu=0, sd=100, shape=3)
    mu_quadratic = beta_quadratic[0] + beta_quadratic[1] * x_1s_cubic[0] + beta_quadratic[2] * x_1s_cubic[0]**2
    sigma_quadratic = pm.HalfCauchy('sigma_quadratic', beta=10, testval=1.0)
    obs_quadratic = pm.Normal('obs_quadratic', mu=mu_quadratic, sd=sigma_quadratic, observed=y_1s_cubic)

    trace_quadratic = pm.sample(2000, tune=1000, cores=1)

# WAIC si LOO pentru fiecare model
waic_linear = az.waic(trace_linear, model_linear)
waic_quadratic = az.waic(trace_quadratic, model_quadratic)
waic_cubic = az.waic(trace_cubic, model_cubic)

loo_linear = az.loo(trace_linear, model_linear)
loo_quadratic = az.loo(trace_quadratic, model_quadratic)
loo_cubic = az.loo(trace_cubic, model_cubic)

print("WAIC - Linear Model:", waic_linear.waic)
print("WAIC - Quadratic Model:", waic_quadratic.waic)
print("WAIC - Cubic Model:", waic_cubic.waic)

print("\nLOO - Linear Model:", loo_linear.loo)
print("LOO - Quadratic Model:", loo_quadratic.loo)
print("LOO - Cubic Model:", loo_cubic.loo)

# afisarea grafica a rezultatelor
az.plot_compare(
    {
        "Linear": trace_linear,
        "Quadratic": trace_quadratic,
        "Cubic": trace_cubic
    },
    ic="waic",
    scale="deviance",
)
plt.show()
