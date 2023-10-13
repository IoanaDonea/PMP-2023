'''
(1pct.) Doi mecanici schimbă filtrele de ulei pentru autoturisme într-un service. Timpul de servire este exponenţial
cu parametrul λ1 = 4 hrs−1 în cazul primului mecanic si λ2 = 6 hrs−1 în cazul celui de al doilea. Deoarece al doilea
mecanic este mai rapid, el serveşte de 1.5 ori mai mulţi clienţi decât partenerul său. Astfel când un client ajunge la rând,
probabilitatea de a servit de primul mecanic este 40%. Fie X timpul de servire pentru un client.
Generaţi 10000 de valori pentru X, şi în felul acesta estimaţi media şi deviaţia standard a lui X. Realizaţi un grafic al
densităţii distribuţiei lui X.
Notă: Distribuţia Exp(λ) se poate apela cu stats.expon(0,1/λ) sau stats.expon(scale=1/λ).
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

lambda1 = 4  # pt primul mecanic
lambda2 = 6  # pt al doilea mecanic

# probabilitatea ca un client să fie servit de primul mecanic
probabilitate_primul_mecanic = 0.4

# numarul total de clienti pentru care dorim sa generam valori
numar_clienti = 10000

# generam aleator valori pt mecanicul care il va servi pe fiecare client
mecanici = np.random.choice([1, 2], numar_clienti, p=[probabilitate_primul_mecanic, 1 - probabilitate_primul_mecanic])

# generam timpul de servire pentru fiecare client
timp_servire = np.zeros(numar_clienti)
for i in range(numar_clienti):
    if mecanici[i] == 1:
        timp_servire[i] = expon(scale=1/lambda1).rvs()
    else:
        timp_servire[i] = expon(scale=1/lambda2).rvs()

# calculam media si deviatia standard a timpului de servire
media = np.mean(timp_servire)
deviatia_standard = np.std(timp_servire)

print(f"Media lui X: {media}")
print(f"Deviația standard a lui X: {deviatia_standard}")

# cream un grafic al densitatii distributiei lui X
plt.hist(timp_servire, bins=50, density=True, alpha=0.6, color='b', label='Distributie timp de servire')
x = np.linspace(0, max(timp_servire), 100)
pdf_primul_mecanic = expon.pdf(x, scale=1/lambda1)
pdf_al_doilea_mecanic = expon.pdf(x, scale=1/lambda2)
pdf_total = probabilitate_primul_mecanic * pdf_primul_mecanic + (1 - probabilitate_primul_mecanic) * pdf_al_doilea_mecanic
plt.plot(x, pdf_total, 'r', linewidth=2, label='Distributie totala')
plt.legend()
plt.xlabel('Timp de servire (X)')
plt.ylabel('Densitate')
plt.title('Distributie a timpului de servire X')
plt.show()

