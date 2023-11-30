import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('Prices.csv')

# Extrage variabilele independente
processor_frequency = data['Speed'].values
log_hard_disk_size = np.log(data['HardDrive'].values)

# Normalizare datelor
processor_frequency = (processor_frequency - processor_frequency.mean()) / processor_frequency.std()
log_hard_disk_size = (log_hard_disk_size - log_hard_disk_size.mean()) / log_hard_disk_size.std()

X = np.vstack((np.ones_like(processor_frequency), processor_frequency, log_hard_disk_size)).T

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=1.0e-2)
    beta = pm.Normal('beta', mu=0, sigma=1.0e-2, shape=2)
    sigma = pm.Uniform('sigma', lower=0, upper=10)

    mu = pm.Deterministic('mu', alpha + pm.math.dot(X, beta))

    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'].values)

    # Simularea distributiei a posteriori
    trace = pm.sample(2000, tune=1000, progressbar=True)

# 2. Estimari HDI pentru β1 și β2
hdi_beta1 = pm.stats.hpd(trace['beta'][:, 0], hdi_prob=0.95)
hdi_beta2 = pm.stats.hpd(trace['beta'][:, 1], hdi_prob=0.95)

print(f"Estimare HDI pentru beta1: {hdi_beta1}")
print(f"Estimare HDI pentru beta2: {hdi_beta2}")

# 3. Verificare intervale de incredere
if 0 not in hdi_beta1 and 0 not in hdi_beta2:
    print("Frecvența procesorului și mărimea hard diskului sunt predictori utili ai prețului de vânzare.")
else:
    print("Frecvența procesorului și/sau mărimea hard diskului nu sunt predictori semnificativi ai prețului de vânzare.")

# 4. Simulare 5000 de extrageri din μ
posterior_mu = trace['mu']
simulated_prices = np.random.choice(posterior_mu, size=5000)

# Construirea intervalului HDI pentru prețul de vanzare simulat
hdi_simulated_prices = pm.stats.hpd(simulated_prices, hdi_prob=0.9)

print(f"Intervalul de 90% HDI pentru prețul de vânzare simulat: {hdi_simulated_prices}")

# 5. Simulare 5000 de extrageri din distributia predictiva posterioara
posterior_predictive = pm.sample_posterior_predictive(trace, samples=5000, model=model)

simulated_prices_predictive = posterior_predictive['y']

# Construirea intervalului HDI pentru prețul de vanzare simulat
hdi_simulated_prices_predictive = pm.stats.hpd(simulated_prices_predictive, hdi_prob=0.9)

print(f"Intervalul de 90% HDI pentru prețul de vânzare simulat (predictiv): {hdi_simulated_prices_predictive}")
