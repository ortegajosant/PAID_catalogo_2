import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal


def convolucion_matricial(A, B):
    """
        Se realiza la convolución en 2D
        Parametros:
        A : es una matriz de mxn que representa una imagen
        B : es un matriz de pxq que representa el kernel
    """

    A = np.array(A)
    B = np.array(B)

    m_1, n_1 = A.shape
    m_2, n_2 = B.shape

    colum = m_1 + m_2 - 1
    row = n_1 + n_2 - 1

    # Matriz resultado
    C = np.zeros((colum, row))

    for j in range(colum):
        for k in range(row):
            p = max(0, j - m_2)
            s_1_f = min(j, m_1)
            while p < s_1_f:
                q = max(0, j - n_2)
                s_2_f = min(j, n_1)
                while q < s_2_f:
                    if 0 <= p <= m_1 and 0 <= q <= n_1 and 0 <= j - p < m_2 and 0 <= k - q < n_2:
                        value = A[p][q] * B[j - p][k - q]
                        C[j][k] = value
                    q += 1
                p += 1
    return C


# Probando la convolución.
A = [[1, 0, 1],
     [4, 3, 1],
     [-1, 3, 1],
     [3, 0, -7]]

B = [[1, -1, 2, 3],
     [-4, 0, 1, 5],
     [3, 2, -1, 0]]

C = convolucion_matricial(A, B)
C2 = signal.convolve2d(A, B, mode='full')

plt.figure()
plt.subplot(131)
plt.title("Matriz original")
plt.imshow(A, cmap='gray')
plt.subplot(132)
plt.title("Convolucion manual")
plt.imshow(C, cmap='gray')
plt.subplot(133)
plt.title("Convolución Python")
plt.imshow(C2, cmap='gray')
plt.show()
