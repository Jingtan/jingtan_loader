# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


x_1 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
x_2 = np.linspace(0.1, 1, 256, endpoint=True)
y_2 = 3/np.tanh((3*x_2))

y_1 = [3.0713,  3.0772,  3.0911,  3.1169,  3.1617,  3.2389,  3.3764,  3.6397,  4.2073,  5.6309]

plt.plot(x_2, y_2, label='True metric')
plt.plot(x_1, y_1, label='Emergent metric')
plt.legend()
plt.show()
