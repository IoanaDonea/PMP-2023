import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

# a.
df = pd.read_csv('auto-mpg.csv')
df_cleaned = df.dropna()

plt.scatter(df_cleaned['horsepower'], df_cleaned['mpg'])
plt.xlabel('CP ')
plt.ylabel('mpg (mile/galon)')
plt.title('Relația dintre CP și mpg')
plt.show()

# b.
with pm.Model():
    alpha = pm.Normal('alpha', mu=0, tau=0.01)
    beta = pm.Normal('beta', mu=0, tau=0.01)
    sigma = pm.Uniform('sigma', lower=0, upper=10)

    @pm.deterministic
    def linear_model(x=df_cleaned['horsepower'].values, alpha=alpha, beta=beta):
        return alpha + beta * x

    obs = pm.Normal('obs', mu=linear_model, tau=1.0/sigma**2, value=df_cleaned['mpg'].values, observed=True)

    mcmc = pm.MCMC([obs, alpha, beta, sigma])
    mcmc.sample(iter=10000, burn=1000)

    alpha_samples = mcmc.trace('alpha')[:]
    beta_samples = mcmc.trace('beta')[:]

    x_values = np.linspace(df_cleaned['horsepower'].min(), df_cleaned['horsepower'].max(), 100)
    y_values = alpha_samples.mean() + beta_samples.mean() * x_values

    # c.
    plt.plot(x_values, y_values, label='Linie de regresie')

    # d.
    y_hdi = pm.hpd(linear_model(x_values), alpha=0.05)
    plt.fill_between(x_values, y_hdi[:, 0], y_hdi[:, 1], color='gray', alpha=0.3, label='95%HDI')

    plt.legend()
    plt.show()
