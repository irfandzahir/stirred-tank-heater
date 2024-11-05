import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function to define the system of ODEs
def system_odes(y, t, F_i, F_j, T_i, T_j_i, U_A, V, V_j, rho_Cp):
    T, T_j = y
    # Constants
    UA = U_A
    rho_Cp_tank = rho_Cp
    rho_Cp_jacket = rho_Cp

    # ODEs
    dTdt = (F_i / V) * (T_i - T) + (UA / (rho_Cp_tank * V)) * (T_j - T)
    dT_jdt = (F_j / V_j) * (T_j_i - T_j) - (UA / (rho_Cp_jacket * V_j)) * (T_j - T)
    return [dTdt, dT_jdt]

# Streamlit app
st.title("Tank and Jacket Temperature Simulation")
st.sidebar.header("Input Variables")

# Sidebar inputs for disturbance and manipulated variables
F_i = st.sidebar.slider("Tank Inlet Flow Rate (F_i) [ft³/min]", 0.5, 2.0, 1.0, step=0.1)
F_j = st.sidebar.slider("Jacket Flow Rate (F_j) [ft³/min]", 0.1, 3.0, 1.5, step=0.1)
T_i = st.sidebar.slider("Inlet Temperature of Tank Fluid (T_i) [°F]", 30.0, 100.0, 50.0, step=1.0)
T_j_i = st.sidebar.slider("Inlet Temperature of Jacket Fluid (T_j_i) [°F]", 150.0, 250.0, 200.0, step=1.0)
U_A = st.sidebar.slider("Heat Transfer Coefficient (U*A) [Btu/°F min]", 100.0, 300.0, 183.9, step=1.0)
V = st.sidebar.slider("Volume of Tank (V) [ft³]", 5.0, 20.0, 10.0, step=1.0)
V_j = st.sidebar.slider("Volume of Jacket (V_j) [ft³]", 0.5, 5.0, 1.0, step=0.1)
rho_Cp = st.sidebar.slider("Heat Capacity (rho*Cp) [Btu/°F ft³]", 50.0, 100.0, 61.3, step=1.0)

# Initial conditions and old steady-state values
T_initial = 125.0  # °F (Tank temperature)
T_j_initial = 150.0  # °F (Jacket temperature)
y0 = [T_initial, T_j_initial]

# Time points
t = np.linspace(0, 100, 1000)  # Simulate for 100 minutes

# Solve ODEs
solution = odeint(system_odes, y0, t, args=(F_i, F_j, T_i, T_j_i, U_A, V, V_j, rho_Cp))
T = solution[:, 0]
T_j = solution[:, 1]

# Calculate new steady-state values (last value in the simulation)
T_new_steady = T[-1]
T_j_new_steady = T_j[-1]

# Plot Tank Temperature
fig1, ax1 = plt.subplots()
ax1.plot(t, T, label="Tank Temperature (°F)", color='blue')
ax1.axhline(y=T_initial, color='green', linestyle='--', label=f"Old Steady-State: {T_initial}°F")
ax1.axhline(y=T_new_steady, color='orange', linestyle='--', label=f"New Steady-State: {T_new_steady:.2f}°F")
ax1.set_xlabel("Time (minutes)")
ax1.set_ylabel("Temperature (°F)")
ax1.set_title("Tank Temperature Over Time")
ax1.legend()
ax1.grid()

# Plot Jacket Temperature
fig2, ax2 = plt.subplots()
ax2.plot(t, T_j, label="Jacket Temperature (°F)", color='red')
ax2.axhline(y=T_j_initial, color='green', linestyle='--', label=f"Old Steady-State: {T_j_initial}°F")
ax2.axhline(y=T_j_new_steady, color='orange', linestyle='--', label=f"New Steady-State: {T_j_new_steady:.2f}°F")
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("Temperature (°F)")
ax2.set_title("Jacket Temperature Over Time")
ax2.legend()
ax2.grid()

# Display the plots in Streamlit
st.pyplot(fig1)
st.pyplot(fig2)
