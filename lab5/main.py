import pymc3 as pm
import numpy as np

data = np.array([5, 6, 7, 8, 10, 12, 14, 15, 17, 19, 21, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])

# intervalul orar
time_intervals = np.arange(4 * 60, 24 * 60, 1)

# definirea modelului probabilistic
with pm.Model() as traffic_model:
    λ = pm.Gamma("λ", alpha=1, beta=1)

    # distributia Poisson pentru valorile de trafic
    traffic = pm.Poisson("traffic", mu=λ, observed=data)

    # crearea intervalelor de timp pentru modificari
    change_intervals = [7 * 60, 8 * 60, 16 * 60, 19 * 60]

    intervals = []
    for i in range(len(change_intervals) - 1):
        interval_name = f"interval_{i + 1}"
        interval_start = change_intervals[i]
        interval_end = change_intervals[i + 1]

        interval = pm.Deterministic(interval_name, pm.math.switch(
            (interval_start <= time_intervals) * (time_intervals < interval_end),
            λ,
            0
        ).sum())

        intervals.append(interval)

with traffic_model:
    trace = pm.sample(10000, tune=2000)

pm.summary(trace)
