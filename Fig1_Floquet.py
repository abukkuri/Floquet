import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt

sigJ = 0.1
sigA = 0.1
gam = 0.1
fluct = 1

def beta(t):
    return fluct * np.sin(2 * np.pi * t) + fluct

# Function to calculate the dominant Floquet exponent
def calculate_dominant_floquet_exponent(sigJ, sigA, gam, v):
    def ecol_system(t, z):
        J, A = z
        dJdt = beta(t) * A * (v - 1) - sigJ * v**2 * J - gam * J
        dAdt = gam * J - sigA * A
        return [dJdt, dAdt]

    T = 1
    t_points = np.linspace(0, T, 100)

    initial_conditions = [1.0, 1.0]
    sol = solve_ivp(ecol_system, [0, T], initial_conditions, t_eval=t_points, method='RK45')

    monodromy_matrix = np.eye(2)
    for i in range(len(t_points) - 1):
        t = t_points[i]
        delta_t = t_points[i + 1] - t_points[i]
        sol_i = sol.y[:, i]
        sol_i1 = sol.y[:, i + 1]
        monodromy_matrix = np.dot(expm(np.array([[-sigJ * v**2 - gam, beta(t) * (v - 1)],[gam, -sigA ]]) * delta_t), monodromy_matrix)

    eigenvalues, _ = np.linalg.eig(monodromy_matrix)

    small_constant = 1e-10
    floquet_exponents = np.log(np.abs(eigenvalues) + small_constant) / T
    dominant_exponent = max(floquet_exponents)

    return dominant_exponent

# Function to calculate the derivative of the dominant Floquet exponent with respect to v
def calculate_dominant_floquet_derivative(v, perturbation):
    v_plus = v + perturbation
    v_minus = v - perturbation
    dominant_plus = calculate_dominant_floquet_exponent(sigJ, sigA, gam, v_plus)
    dominant_minus = calculate_dominant_floquet_exponent(sigJ, sigA, gam, v_minus)
    return (dominant_plus - dominant_minus) / (2 * perturbation)

# Function to define the system of ODEs
def system(t, z, k, perturbation):
    J, A, v = z
    dJdt = beta(t) * A * (v - 1) - sigJ * v**2 * J - gam * J
    dAdt = gam * J - sigA * A
    dvdt = k * calculate_dominant_floquet_derivative(v, perturbation)
    return [dJdt, dAdt, dvdt]

# Set initial values
initial_v = 1
initial_J = 1
initial_A = 1

# Simulation time
T = 80 # Total simulation time
n_steps = T*10  # Number of time steps
t_points = np.linspace(0, T, n_steps)

# Perturbation for calculating the derivative
perturbation = 0.001

k = 2

# Initial conditions
initial_conditions = [initial_J, initial_A, initial_v]

# Integrate the system over time
sol = solve_ivp(system, [0, T], initial_conditions, args=(k, perturbation), t_eval=t_points)

# Extract the results
J_values = sol.y[0]
A_values = sol.y[1]
v_values = sol.y[2]

# Create a plot of the population dynamics and the evolution of 'v'
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_points, J_values, label="Juvenile", color='b', lw=3)
plt.plot(t_points, A_values, label="Adult", color='r', lw=3)
plt.xlabel("Time")
plt.ylabel("Population Size")
plt.title("Population Dynamics: Floquet")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_points, v_values, color='k', lw=3)
plt.xlabel("Time")
plt.ylabel("v")
plt.title("Life History Strategy Dynamics: Floquet")

plt.tight_layout()
plt.show()

print(J_values[-1])
print(A_values[-1])
print(v_values[-1])
