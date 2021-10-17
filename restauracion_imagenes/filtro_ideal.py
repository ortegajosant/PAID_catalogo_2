from PIL import Image  # Imagenes
import matplotlib.pyplot as plt  # Graficos
import numpy as np  # Libreria matematica


def llenado_cuadrantes(A):
    """
    Replica el cuadrante superior izquierdo de la matriz A
    para los demás cuadrantes
    """
    m, n = A.shape
    for i in range(int(m/2)):
        for j in range(int(n/2)):
            A[m-i-1, n-j-1] = A[i, j]
            A[m-i-1, j] = A[i, j]
            A[i, n-j-1] = A[i, j]
    return A


def filtro_ideal_rechaza_banda(A):

    F_A = np.fft.fftshift(np.fft.fft2(A))

    m, n = A.shape

    Dist = np.zeros([m, n])

    for u in range(m):
        for v in range(n):
            Dist[u, v] = np.sqrt(u**2+v**2)
    D0 = 32
    W = 8
    H = np.ones([m, n])
    ind = np.logical_and(D0-W/2 < Dist, D0+W/2 >= Dist)
    H[ind] = 0
    F_B = llenado_cuadrantes(H)
    F_B = np.fft.fftshift(F_B)

    F_C = F_A*F_B

    F_C = np.fft.fftshift(F_C)
    A_t = np.fft.ifft2(F_C)

    return A_t, F_B, F_C


A = Image.open("ruido_periodico.jpg", "r")
A = A.convert("L")

B = Image.open("camarografo.jpg", "r")
A = A.convert("L")

alpha = 0.35
A = np.array(A, dtype=np.double)
B = np.array(B, dtype=np.double)

C = A*alpha + B

A_t, F_B, F_C = filtro_ideal_rechaza_banda(C)
plt.figure()

plt.subplot(221), plt.title("Imagen original")
plt.imshow(C, cmap='gray')

plt.subplot(222), plt.title("Filtro ideal rechaza banda")
plt.imshow(np.log(1 + np.abs(F_B)), cmap='gray')

plt.subplot(223), plt.title("Convolución")
plt.imshow(np.log(1 + np.abs(np.fft.fftshift(F_C))), cmap='gray')

plt.subplot(224), plt.title("Imagen transformada inversa")
plt.imshow(np.uint8(np.abs(A_t)), cmap='gray')
plt.show()
