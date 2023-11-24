import random
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1.
# Simuleaza aruncarea monedei
# Primeste un argument optional masluita, indicand daca moneda este trucata (implicit este False).
# Returnează 1 (stema) sau 0 (pajura) in functie de rezutatul aruncarii monedei trucate sau netrucate.
def arunca_moneda(masluita=False):
    if masluita:
        # Daca moneda este masluita, avem o probabilitate de 1/3 pentru stema 1 si 2/3 pentru stema 0
        return random.choices([0, 1], weights=[2/3, 1/3])[0]
    else:
        # Daca moneda nu este masluita, avem o probabilitate de 1/2 pentru stema 1 si 1/2 pentru stema 0
        return random.choice([0, 1])


# Simuleaza jocul (fara retea bayesiana)
#Decide aleatoriu care jucător începe.
# Simuleaza aruncarea monedei pentru primul jucator (steme_j0).
# Simuleaza aruncarea monedei pentru al doilea jucator (steme_j1) în functie de rezultatul aruncarii primului jucator.
def simuleaza_joc():
    # Decide cine incepe
    jucator_initial = random.choice([0, 1])
    # Arunca moneda pentru jucatorul initial
    steme_j0 = arunca_moneda(jucator_initial == 0)

    # Jucatorul opus arunca moneda de n+1 ori in a doua runda
    for _ in range(steme_j0 + 1):
        steme_j1 = arunca_moneda(jucator_initial == 1)

    return steme_j0, steme_j1


# Simuleaza 20000 de jocuri si afiseaza rezultatele
def simulare_multipla_si_afisare(numar_jocuri):
    castiguri_j0 = 0
    castiguri_j1 = 0

    # Simuleaza un numar dat de jocuri
    for _ in range(numar_jocuri):
        steme_j0, steme_j1 = simuleaza_joc()

        # Determina castigatorul
        if steme_j0 >= steme_j1:
            castiguri_j0 += 1
        else:
            castiguri_j1 += 1

    # Calculeaza procentajele de castig
    procentaj_castig_j0 = (castiguri_j0 / numar_jocuri) * 100
    procentaj_castig_j1 = (castiguri_j1 / numar_jocuri) * 100

    # Afiseaza rezultatele
    print(f"Jucatorul J0 a castigat {procentaj_castig_j0}% din jocuri.")
    print(f"Jucatorul J1 a castigat {procentaj_castig_j1}% din jocuri.")


#2.
# Crearea modelului Bayesian
model = BayesianNetwork([('Moneda_masluita', 'Steme_J0'), ('Moneda_normala', 'Steme_J1'), ('J0_starts', 'Steme_J0'), ('J1_starts', 'Steme_J1')])

# Adaugarea CPDs la model
cpd_moneda_masluita = TabularCPD('Moneda_masluita', 2, values=[[0.5], [0.5]])
cpd_moneda_normala = TabularCPD('Moneda_normala', 2, values=[[0.5], [0.5]])
cpd_j0_starts = TabularCPD('J0_starts', 2, values=[[0.5], [0.5]])
cpd_j1_starts = TabularCPD('J1_starts', 2, values=[[0.5], [0.5]])

model.add_cpds(cpd_moneda_masluita, cpd_moneda_normala, cpd_j0_starts, cpd_j1_starts)


# Functia care simuleaza un joc cu retea bayesiana
# Decide aleatoriu care jucator incepe.
# Seteaza evidente pentru model in functie de rezultatul aruncarii monedelor.
# Foloseste model.predict_proba pentru a obtine probabilitatile stemei pentru ambii jucatori.
def simuleaza_joc_pgmpy(model):
    prob_J0_starts = 0.5
    jucator_initial = 'J0_starts' if random.random() < prob_J0_starts else 'J1_starts'
    model.nodes[jucator_initial]['evidence'] = True

    model.nodes['Steme_J0']['evidence'] = arunca_moneda(model.nodes['Moneda_masluita']['evidence'])
    model.nodes['Steme_J1']['evidence'] = arunca_moneda(model.nodes['Moneda_normala']['evidence'])

    steme_j0 = model.predict_proba(variables=['Steme_J0'])[0].values[1]
    steme_j1 = model.predict_proba(variables=['Steme_J1'])[0].values[1]

    print(f"Jucatorul {jucator_initial} a castigat jocul cu {steme_j0} steme impotriva lui J1 cu {steme_j1} steme.")
    return steme_j0, steme_j1

# Functia care simuleaza mai multe jocuri cu retea bayesiana si afiseaza rezultatele
def simulare_multipla_pgmpy(numar_jocuri):
    castiguri_j0 = 0
    castiguri_j1 = 0

    for _ in range(numar_jocuri):
        steme_j0, steme_j1 = simuleaza_joc_pgmpy(model)

        if steme_j0 >= steme_j1:
            castiguri_j0 += 1
        else:
            castiguri_j1 += 1

    procentaj_castig_j0 = (castiguri_j0 / numar_jocuri) * 100
    procentaj_castig_j1 = (castiguri_j1 / numar_jocuri) * 100

    print(f"Jucatorul J0 a castigat {procentaj_castig_j0}% din jocuri.")
    print(f"Jucatorul J1 a castigat {procentaj_castig_j1}% din jocuri.")

#3.
# Functia care determina ce fata a monedei este cea mai probabil sa se fi obtinut in prima runda
# Această functie are rolul de a efectua inferența Bayesiana pentru a determina probabilitatile asociate fetei
# monedei in prima runda, avand in vedere starea variabilelor 'Steme_J0' si 'Steme_J1' resetate la None.
def determina_probabil_fata_monedei_in_prima_runda(model):
    # Reseteaza evidenta pentru nodurile de steme
    model.nodes['Steme_J0']['evidence'] = None
    model.nodes['Steme_J1']['evidence'] = None

    # Realizeaza inferenta Bayesiana pentru a determina probabilitatile aferente fetei monedei in prima runda
    inferenta = VariableElimination(model)  # crearea unui obiect VariableElimination pentru reteaua Bayesiana (model), care este utilizat ulterior pentru a efectua inferenta si a obtine distributiile de probabilitate
    probabilitati_fata_monedei = inferenta.query(variables=['Moneda_masluita', 'Moneda_normala'], joint=True)

    print("Probabilitati pentru fata monedei in prima runda:")
    print(probabilitati_fata_monedei)



numar_jocuri = 20000
simulare_multipla_si_afisare(numar_jocuri)
simulare_multipla_pgmpy(numar_jocuri)
determina_probabil_fata_monedei_in_prima_runda(model)