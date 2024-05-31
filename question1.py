import numpy as np
import matplotlib.pyplot as plt

#a
N = 200000
n = 20
samples = np.random.binomial(1, 0.5, (N, n))
empirical_means = np.mean(samples, axis=1)

#b
epsilons = np.linspace(0, 1, 50)
empirical_probabilities = [(np.abs(empirical_means - 0.5) > epsilon).mean() for epsilon in epsilons]
plt.plot(epsilons, empirical_probabilities, label='Empirical Probability')
plt.xlabel('Epsilon')
plt.ylabel('Probability')
plt.legend()
plt.show()

#c
hoeffding_bound = [2 * np.exp(-2 * epsilon**2 * n) for epsilon in epsilons]
plt.plot(epsilons, empirical_probabilities, label='Empirical Probability')
plt.plot(epsilons, hoeffding_bound, label='Hoeffding Bound',)
plt.xlabel('Epsilon')
plt.ylabel('Probability')
plt.legend()
plt.show()