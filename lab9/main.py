import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Admission.csv')

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, tau=1.0/10**2)
    beta1 = pm.Normal('beta1', mu=0, tau=1.0/10**2)
    beta2 = pm.Normal('beta2', mu=0, tau=1.0/10**2)

    # probabilitatea de admitere pentru fiecare student
    pi = pm.invlogit(beta0 + beta1 * data['GRE'] + beta2 * data['GPA'])

    # rezultatele de admitere
    admission = pm.Bernoulli('admission', p=pi, observed=data['Admission'])

# antreneaza modelul folosind MCMC
with logistic_model:
    trace = pm.sample(draws=10000, tune=1000, chains=1, progressbar=True)

pm.summary(trace).round(2)

# vizualizare distributii marginale
pm.plot_posterior(trace, figsize=(10, 6))
plt.show()


# obtinerea datelor de la esantionul MCMC
trace_df = pm.trace_to_dataframe(trace)

# 2.
# calcularea granitei de decizie si intervalului HDI
decision_boundary = -trace_df['beta0'].mean() / trace_df['beta2'].mean()
hdi_lower, hdi_upper = np.percentile(trace_df['beta1'] * data['GRE'] + trace_df['beta2'] * data['GPA'], [2.5, 97.5])

# vizualizare date si granita de decizie
plt.scatter(data['GRE'][data['Admission'] == 1], data['GPA'][data['Admission'] == 1], marker='o', label='Admis')
plt.scatter(data['GRE'][data['Admission'] == 0], data['GPA'][data['Admission'] == 0], marker='x', label='Respins')

# granita de decizie
plt.plot([data['GRE'].min(), data['GRE'].max()], [decision_boundary, decision_boundary], color='black', linestyle='--', label='Granița de decizie medie')

# intervalul HDI
plt.fill_betweenx(y=[data['GPA'].min(), data['GPA'].max()], x1=hdi_lower, x2=hdi_upper, color='gray', alpha=0.5, label='Interval 94% HDI')

plt.xlabel('GRE Score')
plt.ylabel('GPA')
plt.legend()
plt.show()


# 3.
student_GRE = 550
student_GPA = 3.5
student_pi = pm.invlogit(trace_df['beta0'] + trace_df['beta1'] * student_GRE + trace_df['beta2'] * student_GPA)
student_hdi = np.percentile(student_pi, [5, 95])

print(f'Intervalul HDI pentru probabilitatea admiterii pentru studentul cu scor GRE {student_GRE} și GPA {student_GPA}: {student_hdi}')

# 4.
new_student_GRE = 500
new_student_GPA = 3.2
new_student_pi = pm.invlogit(trace_df['beta0'] + trace_df['beta1'] * new_student_GRE + trace_df['beta2'] * new_student_GPA)
new_student_hdi = np.percentile(new_student_pi, [5, 95])

print(f'Intervalul HDI pentru probabilitatea admiterii pentru studentul cu scor GRE {new_student_GRE} și GPA {new_student_GPA}: {new_student_hdi}')

# Diferenta in probabilitatea de admitere pentru acest nou student poate fi justificata prin contributia relativa a scorului GRE si GPA
# la modelul logistic. In cazul unui scor GRE mai mic si a unui GPA mai mic, probabilitatea de admitere poate scadea.
# Modul in care acesti factori contribuie la probabilitatea de admitere este capturat de distributia posterioara a parametrilor modelului logistic.