import numpy as np
from scipy.stats import poisson

def simulate_fast_food( lmbda=20, media_normala=2.0, deviatia_standard_normala=0.5, alpha=3.0):
    numar_clienti = poisson.rvs(lmbda) 
    timp_plasare_plata = np.random.normal(media_normala, deviatia_standard_normala, size=numar_clienti)

    timp_gatit = np.random.exponential(scale=alpha, size=numar_clienti)



    print("Numarul total de clienti intr-o ora:",  numar_clienti)
    print("Timpul mediu de plasare si plata a comenzii:",timp_plasare_plata )
    print("Timpul mediu de gatit:",   timp_gatit )

    return  numar_clienti,  timp_plasare_plata, timp_gatit

simulate_fast_food()