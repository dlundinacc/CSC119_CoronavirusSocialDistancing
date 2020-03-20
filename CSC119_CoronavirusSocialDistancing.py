import matplotlib.pyplot as plt
import numpy as np

##################################################################################################################
#
# https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
#
# https://www.datahubbs.com/modeling-an-epidemic/
#
##################################################################################################################


def base_seir_model(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1]) * dt
        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1]) * dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1]) * dt
        next_R = R[-1] + (gamma*I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T


def seir_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (rho * beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (rho * beta * S[-1] * I[-1] - alpha * E[-1]) * dt
        next_I = I[-1] + (alpha * E[-1] - gamma * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T


# Define parameters
t_max = 100                                                     # 100 days
dt = .1                                                         # Step-size is 1 day
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000                                                       # Quantity of people in our population
init_vals = 1 - 1/N, 1/N, 0, 0
alpha = 0.2                                                     # Inverse of the incubation period (1/t_incubation)
beta = 1.75                                                     # Average contact rate in the population
gamma = 0.5                                                     # Inverse of the mean infectious period (1/t_infectious)


# Run simulation for SEIR model
params = alpha, beta, gamma
results = base_seir_model(init_vals, params, t)

# Plot results for Exposed and Infected
plt.figure(figsize=(12, 8))
plt.plot(t, results[:, 1], t, results[:, 2])                   # Print column 1 which is Exposed and 2 which is Infected
plt.legend(['Exposed', 'Infected'])
plt.xlabel('Time (Days)')
plt.ylabel('Population Fraction')
plt.title(r'COVID-19 SEIR Model with $\alpha={}, \beta={}, \gamma={}$'.format(params[0], params[1], params[2]))

fname = r'COVID-19 SEIR Model with alpha={} beta={} gamma={}.png'.format(params[0],params[1],params[2])
print("Writing plot to filename="+fname)
plt.grid(b=True, which='major', axis='both')
plt.savefig(fname)

plt.show()



# Run simulation for SEIR with Social Distancing model

# Repeat simulation with rho = 1.0, 0.8 and 0.5

plt.figure(1, figsize=(12, 8))
plt.xlabel('Time (Days)')
plt.ylabel('Population Fraction')
plt.title(
    r'COVID-19 SEIR Model with Social Distancing with $\alpha={}, \beta={}, \gamma={}$'.format(params[0], params[1],
                                                                                               params[2]))

for i in range (0, 3):

    if (i==0):
        rho = 1.0
        color = 'b'
    elif (i==1):
        rho = 0.8
        color = 'g'
    else:
        rho = 0.5
        color = 'r'

    params = alpha, beta, gamma, rho
    results = seir_model_with_soc_dist(init_vals, params, t)

    # Plot results
    legendExposed = r'Exposed $\rho={}$'.format(rho)
    legendInfected = r'Infected $\rho={}$'.format(rho)
    plt.legend([legendExposed, legendInfected])
    plt.grid(b=True, which='major', axis='both')
    plt.plot(t, results[:, 1], color, t, results[:, 2], color+'--')         # Print column 1 which is Exposed and 2 which is Infected

fname = r'COVID-19 SEIR Model with Social Distancing with alpha_{} beta_{} gamma_{}.png'.format(params[0], params[1],
                                                                                                params[2])
print("Writing plot to filename=" + fname)

plt.savefig(fname)
plt.show()
