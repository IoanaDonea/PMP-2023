import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns

clusters = 3
n_cluster = [200, 150, 100]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
                       np.repeat(std_devs, n_cluster))

mix = mix.reshape(-1, 1)

for n_components in [2, 3, 4]:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(mix)

    sns.kdeplot(data=mix, color='blue', label='Data', fill=True)
    plt.title(f'Model de Mixtura de {n_components} Distr. Gaussiene')

    for i in range(n_components):
        mu = gmm.means_[i][0]
        sigma = np.sqrt(gmm.covariances_[i][0, 0])
        weight = gmm.weights_[i]
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = weight * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        plt.plot(x, y, label=f'Component {i + 1}')

    plt.legend()
    plt.show()
