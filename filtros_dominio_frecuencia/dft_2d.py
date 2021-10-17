import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

m = 20
n = 30

A = np.random.rand(m, n)

plt.figure()
plt.subplot(131)
plt.title("Matriz original")
plt.imshow(A, cmap='gray')

# Ading DFT 2D

F1 = np.zeros((m, n))

for u in range(m):
    for v in range(n):
        for x in range(m-1):
            for y in range(n-1):
                F1[u, v] += A[x, y] * \
                    np.exp(np.complex(0, -2*np.pi*((u*x)/m+(v*y)/n)))


plt.subplot(132)
plt.title("Matriz F1")
plt.imshow(np.uint8(F1), cmap='gray')

F2 = np.fft.fft2(A)

plt.subplot(133)
plt.title("Matriz F2")
plt.imshow(np.uint8(F2), cmap='gray')

plt.show()
