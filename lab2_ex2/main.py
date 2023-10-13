'''Patru servere web oferă acelaşi serviciu (web) clienţilor . Timpul necesar procesării unei cereri (request)
HTTP este distribuit Γ(4, 3) pe primul server, Γ(4, 2) pe cel de-al doilea, Γ(5, 2) pe cel de-al treilea, şi Γ(5, 3) pe cel de-al
patrulea (în milisecunde). La această durată se adaugă latenţa dintre client şi serverele pe Internet, care are o distribuţie
exponenţială cu λ = 4 (în miliseconde−1

). Se ştie că un client este direcţionat către primul server cu probabilitatea 0.25,
către al doilea cu probabilitatea 0.25, iar către al treilea server cu probabilitatea 0.30. Estimaţi probabilitatea ca timpul
necesar servirii unui client, notat cu X, (de la lansarea cererii până la primirea răspunsului) să fie mai mare decât 3
milisecunde. Realizaţi un grafic al densităţii distribuţiei lui X.
Notă: Distribuţia Γ(α, λ) se poate apela cu stats.gamma(α,0,1/λ) sau stats.gamma(α,scale=1/λ).'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# parametrii distributiei Gamma pt timpul de procesare pe fiecare server
alpha = [4, 4, 5, 5]
beta = [3, 2, 2, 3]

# parametrul pentru distributia Exponentială pentru latenta
lambda_latenta = 4  # în milisecunde^(-1)

# prob ca un client este directionat catre fiecare server
probabilitati_servere = [0.25, 0.25, 0.30, 0.20]

# nr de esantioane
num_esantioane = 10000

# generarea esantioanelor pt timpul necesar servirii unui client (X)
timp_servire = np.zeros(num_esantioane)
for i in range(num_esantioane):
    server_ales = np.random.choice([0, 1, 2, 3], p=probabilitati_servere)
    timp_procesare = np.random.gamma(alpha[server_ales], scale=1/beta[server_ales])
    latenta = np.random.exponential(scale=1/lambda_latenta)
    timp_servire[i] = timp_procesare + latenta

# estimarea prob ca X > 3 milisecunde
probabilitate_X_mai_mare_de_3 = np.mean(timp_servire > 3)

print(f"Probabilitatea ca X > 3 milisecunde: {probabilitate_X_mai_mare_de_3}")

# crearea unui grafic al densitatii distributiei lui X
plt.hist(timp_servire, bins=70, density=True, alpha=0.5, color='g')
x = np.linspace(0, max(timp_servire), 100)
pdf = stats.expon.pdf(x, scale=1/lambda_latenta)
plt.plot(x, pdf, 'r-', lw=2)
plt.xlabel('X (Timpul de servire)')
plt.ylabel('Densitatea de probabilitate')
plt.title('Densitatea distributiei lui X')
plt.show()
