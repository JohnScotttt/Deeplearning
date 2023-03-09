import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)

def f(x):
    return np.sin(x)


y = f(x)
dy1 = (f(x+0.001)-f(x-0.001))/0.002

plt.subplot(211)
plt.plot(x, y, label='y=sin')
plt.subplot(211)
plt.plot(x, dy1, label="y'")
plt.title('Numerical differentiation')
plt.legend()

x_torch = torch.tensor(x, requires_grad=True)
y2 = torch.sin(x_torch)
y2.sum().backward()
dy2 = x_torch.grad

plt.subplot(212)
plt.plot(x, y, label='y=sin')
plt.subplot(212)
plt.plot(x, dy2, label="y'")
plt.title('Automatic derivation')
plt.legend()

plt.show()
