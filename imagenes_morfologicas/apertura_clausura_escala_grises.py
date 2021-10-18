from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def erosion(A):
    m, n = A.shape
    C = np.zeros((m, n))

# ------------------------------- Esquinas -----------------------------

    # Esquina (0,0)
    Aux = A[0: 1, 0:1]
    C[0, 0] = np.min(np.min(Aux))

    # Esquina (0, n-1)
    Aux = A[0: 1, n-2:n-1]
    C[0, n-1] = np.min(np.min(Aux))

    # Esquina (m-1, 0)
    Aux = A[m-2: m-1, 0:1]
    C[m-1, 0] = np.min(np.min(Aux))

    # Esquina (m-1, n-1)
    Aux = A[m-2: m-1, n-2:n-1]
    C[m-1, n-1] = np.min(np.min(Aux))

# ------------------------------- Bordes -------------------------------

    # Borde superior e inferior
    for x in range(1, n-1):
        # Superior
        # Esquina (m-1, n-1)
        Aux = A[0: 1, x-1:x+1]
        C[0, x] = np.min(np.min(Aux))

        # Inferior
        Aux = A[m-2: m-1, x-1:x+1]
        C[m-1, x] = np.min(np.min(Aux))

   # Borde derecho e izquierdo
    for y in range(1, m-1):
        # Izquierdo
        Aux = A[y-1: y+1, 0:1]
        C[y, 0] = np.min(np.min(Aux))

        # Izquierdo
        Aux = A[y-1: y+1, n-2:n-1]
        C[y, n-1] = np.min(np.min(Aux))

    for x in range(1, m-1):
        for y in range(1, n-1):
            Aux = A[x-1: x+1, y-1:y+1]
            C[x, y] = np.min(np.min(Aux))

    return C


def dilatacion(A):
    m, n = A.shape
    C = np.zeros((m, n))

# ------------------------------- Esquinas -----------------------------

    # Esquina (0,0)
    Aux = A[0: 1, 0:1]
    C[0, 0] = np.min(np.min(Aux))

    # Esquina (0, n-1)
    Aux = A[0: 1, n-2:n-1]
    C[0, n-1] = np.min(np.min(Aux))

    # Esquina (m-1, 0)
    Aux = A[m-2: m-1, 0:1]
    C[m-1, 0] = np.min(np.min(Aux))

    # Esquina (m-1, n-1)
    Aux = A[m-2: m-1, n-2:n-1]
    C[m-1, n-1] = np.min(np.min(Aux))

# ------------------------------- Bordes -------------------------------

    # Borde superior e inferior
    for x in range(1, n-1):
        # Superior
        # Esquina (m-1, n-1)
        Aux = A[0: 1, x-1:x+1]
        C[0, x] = np.min(np.min(Aux))

        # Inferior
        Aux = A[m-2: m-1, x-1:x+1]
        C[m-1, x] = np.min(np.min(Aux))

   # Borde derecho e izquierdo
    for y in range(1, m-1):
        # Izquierdo
        Aux = A[y-1: y+1, 0:1]
        C[y, 0] = np.min(np.min(Aux))

        # Izquierdo
        Aux = A[y-1: y+1, n-2:n-1]
        C[y, n-1] = np.min(np.min(Aux))
    for x in range(1, m-1):
        for y in range(1, n-1):
            Aux = A[x-1: x+1, y-1:y+1]
            C[x, y] = np.max(np.max(Aux))

    return C


def apertura(A):
    B = erosion(A)
    C = dilatacion(B)
    return B - C


def clausura(A):
    B = dilatacion(A)
    C = erosion(B)
    return B - C


def top_hat(A, A_apertura):
    B = A - A_apertura
    return B


def bottom_hat(A, A_clausura):
    B = A_clausura - A
    return B


A = Image.open("imagen11.jpg", "r")
A = np.array(A, dtype=np.uint8)

plt.figure()
plt.subplot(231)
plt.title("Imagen Original")
plt.imshow(A, cmap="gray")

A_apertura = apertura(A)
plt.subplot(232)
plt.title("Imagen Apertura")
plt.imshow(A_apertura, cmap="gray")

A_clausura = clausura(A)
plt.subplot(233)
plt.title("Imagen Clausura")
plt.imshow(A_clausura, cmap="gray")

A_top_hat = top_hat(A, A_apertura)
plt.subplot(234)
plt.title("Imagen Top Hat")
plt.imshow(A_top_hat, cmap="gray")

A_bottom_hat = bottom_hat(A, A_clausura)
plt.subplot(235)
plt.title("Imagen Bottom Hat")
plt.imshow(A_bottom_hat, cmap="gray")

plt.show()
