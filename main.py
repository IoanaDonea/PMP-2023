from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# def var aleatoare
model = BayesianNetwork([('C', 'I'),
                         ('C', 'A'),
                         ('I', 'A')])

# def tabelului
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], #incendiu fara cutremur
                           evidence=['C'], evidence_card=[2])
cpd_a = TabularCPD(variable='A', variable_card=2,
                        values=[[0.999, 0.02, 0.95, 0.1],
                                [0.001, 0.98, 0.05, 0.9]], #A=1/I=1,C=0
                        evidence=['C', 'I'], evidence_card=[2, 2])

# atașarea tabelului CPD la rețea
model.add_cpds(cpd_c, cpd_i, cpd_a)

assert model.check_model()
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

# crearea unui motor de inferență pe baza modelului
inference = VariableElimination(model)

# prob că a avut loc un cutremur,  cu declanșarea alarmei de incendiu
result = inference.query(variables=['C'], evidence={'A': 1})
print(result)

#probabilitatea ca un incendiu sa fi avut loc, fără ca alarma de incendiu să se activeze.
result = inference.query(variables=['I'], evidence={'A': 0})
print(result)
