'''Se consideră un experiment aleator prin aruncarea de 10 ori a două monezi, una nemăsluită, cealaltă cu
probabilitatea de 0.3 de a obţine stemă. Să se genereze 100 de rezultate independente ale acestui experiment şi astfel
să se determine grafic distribuţiile variabilelor aleatoare care numără rezultatele posibile în cele 10 aruncări (câte una
pentru fiecare rezultat posibil: ss, sb, bs, bb).'''

import random
import matplotlib.pyplot as plt

# functie pentru a simula aruncarea monezilor si a returna rezultatul
def arunca_monezi():
    monezi = ['s', 'b']
    rezultat_aruncare = [random.choice(monezi), random.choices(monezi, weights=[0.7, 0.3])[0]]
    return ''.join(rezultat_aruncare)

# generam 100 de rezultate independente ale experimentului
rezultate = [arunca_monezi() for _ in range(100)]

# initializam numaratoarele pt fiecare combinatie de rezultate
numar_ss = 0
numar_sb = 0
numar_bs = 0
numar_bb = 0

# numaram aparitiile fiecărei combinatii posibile de rezultate
for rezultat in rezultate:
    if rezultat == 'ss':
        numar_ss += 1
    elif rezultat == 'sb':
        numar_sb += 1
    elif rezultat == 'bs':
        numar_bs += 1
    elif rezultat == 'bb':
        numar_bb += 1

combinații = ['ss', 'sb', 'bs', 'bb']
numere = [numar_ss, numar_sb, numar_bs, numar_bb]

plt.bar(combinații, numere, color=['blue', 'green', 'violet', 'purple'])
plt.xlabel('Combinatii rezultate')
plt.ylabel('Numar de aparitii')
plt.title('Distributia rezultatelor in 100 de aruncari')
plt.show()
