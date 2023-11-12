import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

pj = 0.1
pa = 0.1
b = 10
trans = 0.1
time = 80
fluct = 1

bl = 1
bu = 6

# Define a range of v values
v_values = np.linspace(bl, bu, 100)  # 0 to 8

final_population_sizes = []

for v in v_values:
    def b(t):
        return fluct * np.sin(2 * np.pi * t) + fluct

    IC = [1, 1]

    def evoLV(X, t):
        J = X[0]
        A = X[1]
        dJdt = b(t) * (v - 1) * A - pj * v**2 * J - trans * J
        dAdt = trans * J - pa * A
        dxvdt = np.array([dJdt, dAdt])
        return dxvdt

    intxv = np.array(IC)
    pop = odeint(evoLV, intxv, range(time + 1))
    final_population_sizes.append(pop[-1, 0] + pop[-1, 1])

# Find the index of the maximum final population size
max_index = np.argmax(final_population_sizes)
max_v = v_values[max_index]

# Format max_v with the desired number of decimal places
max_v_formatted = f"{max_v:.2f}"

# Plot the final population size as a function of v
plt.figure()
plt.title('Final Population Size vs. Life History Strategy')
plt.plot(v_values, final_population_sizes, lw=3, color='k')
plt.axvline(2.68, color='c', linestyle='--',label='Floquet') 
plt.axvline(2.84, color='r',linestyle='--',label='Spectral Bound')
plt.xlabel('v')
plt.ylabel('Final Population Size')
plt.ylim(bottom=0)
plt.xlim(bl,bu)
plt.legend()
plt.show()
