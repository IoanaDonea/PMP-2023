import numpy as np
from scipy.stats import poisson

def simulate_fast_food(lmbda=20, media_normala=2.0, deviatia_standard_normala=0.5, alpha=3.0):
    numar_clienti = poisson.rvs(lmbda)
    timp_plasare_plata = np.random.normal(media_normala, deviatia_standard_normala, size=numar_clienti)
    timp_gatit = np.random.exponential(scale=alpha, size=numar_clienti)

    return numar_clienti, timp_plasare_plata, timp_gatit

def calculate_alpha_for_service_time(target_time, lmbda=20):
    alpha = 0.1

    while True:
        numar_mediu_clienti = lmbda
        timp_gatit_mediu = 1.0 / alpha

        timp_mediu_servire = timp_gatit_mediu
        if timp_mediu_servire < target_time:
            return alpha
        alpha += 0.01

def calculate_average_waiting_time(alpha, lmbda=20):
    numar_clienti, timp_plasare_plata, timp_gatit = simulate_fast_food(lmbda, alpha=alpha)
    timp_total_servire = timp_plasare_plata + timp_gatit
    timp_mediu_asteptare = np.mean(timp_total_servire)
    return timp_mediu_asteptare

if __name__ == "__main__":
    alpha_maxim = calculate_alpha_for_service_time(15)

    print("Alpha maxim pentru timpul mediu de servire mai mic de 15 minute:", alpha_maxim)

    timp_mediu_asteptare_client = calculate_average_waiting_time(alpha_maxim)
    print("Timpul mediu de asteptare al unui client:", timp_mediu_asteptare_client)
