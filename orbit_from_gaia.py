"""
A script to find the position angle and ratio 
(separation / a) for a binary star system given
Gaia orbital parameters.
"""
import numpy as np
from scipy.optimize import fsolve
from astropy.time import Time
import matplotlib.pyplot as plt
plt.ion()

#--- Orbital parameters from Gaia ---
# HD 222805
A_TI = -5.02e-1 #Thiele-Innes parameter A (mas)
B_TI = -7.54e-1 #Thiele-Innes parameter B (mas)
F_TI = -8.41e-1 #Thiele-Innes parameter F (mas)
G_TI = 1.58e-1  #Thiele-Innes parameter G (mas)
e = 0.456       #Eccentricity
P = 215.6696    #Period (days)
T0 = -38.27     #Time of periastron (days since J2016.0)
a_orb = 1.0405     #Semi-major axis (mas)

# HD 5877
A_TI = 1.4879 #Thiele-Innes parameter A (mas)
B_TI = 0.667 #Thiele-Innes parameter B (mas)
F_TI = -0.330 #Thiele-Innes parameter F (mas)
G_TI = 1.350  #Thiele-Innes parameter G (mas)
e = 0.106       #Eccentricity
P = 211.36    #Period (days)
T0 = -81.2     #Time of periastron (days since J2016.0)
a_orb = 1.927    #Semi-major axis (mas


#--- Functions to solve Kepler's equation ---

def kepler_eq(E, M, e):
    """Kepler's equation: M = E - e*sin(E)"""
    return E - e * np.sin(E) - M

def solve_kepler(M, e):
    """Solve Kepler's equation for E given M and e."""
    E0 = M  # Initial guess
    E = fsolve(kepler_eq, E0, args=(M, e))
    return E[0]

#--- Main computation function ---
def compute_binary_parameters(orbital_params):
    """Compute the position angle and separation for a binary star system."""
    # Unpack orbital parameters
    M, e, A, B, F, G = orbital_params

    # Solve for eccentric anomaly
    E = solve_kepler(M, e)

    # Calculate the x and y coordinates in the orbital plane
    x = A * (np.cos(E) - e) + F * np.sqrt(1 - e**2) * np.sin(E)
    y = B * (np.cos(E) - e) + G * np.sqrt(1 - e**2) * np.sin(E)

    # Compute position angle and separation ratio
    # (Placeholder calculations, replace with actual formulas)
    position_angle = np.arctan2(y, x)  
    separation = np.sqrt(x**2 + y**2) 

    return separation, position_angle

if __name__ == "__main__":
    # Example mean anomaly (M) values to compute
    M_values = np.linspace(0, 2 * np.pi, 100)

    sep = []
    pa = []
    for M in M_values:
        orbital_params = (M, e, A_TI, B_TI, F_TI, G_TI)
        s, p = compute_binary_parameters(orbital_params)
        sep.append(s)
        pa.append(p)
    sep = np.array(sep)
    pa = np.array(pa)
    # Compute the current mean anomaly based on time of observation
    obs_jd = Time("2024-09-16T00:00:00", format='isot', scale='utc').jd
    days_since_J2016 = obs_jd - 2457388.5
    M_current = 2 * np.pi * ((days_since_J2016 - T0) % P) / P
    orbital_params_current = (M_current, e, A_TI, B_TI, F_TI, G_TI)
    sep_now, pa_now = compute_binary_parameters(orbital_params_current)
    # Plot the orbit and mark the position at observation time
    plt.figure(5)
    plt.clf()
    plt.plot(sep * 1e3*np.sin(pa), sep * 1e3*np.cos(pa), label='Orbit Path')
    plt.plot(sep_now * 1e3*np.sin(pa_now), sep_now * 1e3*np.cos(pa_now), 'ro', label='Centre of Light')
    plt.xlabel('East Position (µas)')
    plt.ylabel('North Position (µas)')
    plt.title('Binary Star Orbit from Gaia Parameters')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    # Add an "x" at (0,0) to mark the centre of mass
    plt.plot(0, 0, 'kx', markersize=10, label='Centre of Mass')
    plt.legend()
    plt.tight_layout()