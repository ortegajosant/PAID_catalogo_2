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


def filtro_butterworth_alto(A):
    """
    Filtro paso alto butterworth
    """
    A = np.array(A)
    F_A = np.fft.fftshift(np.fft.fft2(A))
    m, n = A.shape
    D0 = 10
    D1 = np.zeros([m, n])
    k = 1  # orden
    F_B = np.zeros([m, n])
    for u in range(m):
        for v in range(n):
            D_uv = np.sqrt(u**2+v**2)
            F_B[u, v] = 1/(1+((D0/D_uv)**(2*k)))
    F_B = llenado_cuadrantes(F_B)
    F_B = np.fft.fftshift(F_B)

    F_C = F_A*F_B

    F_C = np.fft.fftshift(F_C)

    A_t = np.fft.ifft2(F_C)

    return A_t, F_A, F_C


img = "edificio_china.jpg"

A = Image.open(img, "r")
A = A.convert("L")

A_t, F_B, F_C = filtro_butterworth_alto(A)
plt.figure()
# Imagen original
plt.subplot(221), plt.title("Imagen original")
plt.imshow(A, cmap='gray')

plt.subplot(222), plt.title("DFT2 de A")
plt.imshow(np.log(1 + np.abs(F_B)), cmap='gray')

plt.subplot(223), plt.title("Resultado de la convolucion")
plt.imshow(np.log(1 + np.abs(np.fft.fftshift(F_C))), cmap='gray')

plt.subplot(224), plt.title("DFT2 Inversa con filtro paso alto")
plt.imshow(np.uint8(np.abs(A_t)), cmap='gray')
plt.show()
