import numpy as np
import matplotlib.pyplot as plt

nets = []
for i in range(0, 50):
    if i<=20:
        net = i-i*0.13
    else:
        net = i-i*0.3
    nets.append(net)

plt.plot(nets)
plt.grid()
plt.show()
print()