from PIL import Image
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


def to_bin(X):
    m, n = X.shape
    Y = np.zeros((m, n))
    Y[X >= 127] = 255
    Y[X < 127] = 0
    return Y


def complement(A):
    A = to_bin(A)
    A = 255 - A
    return A


def rellenar(A, B, iters=100, do_plot=False):
    m, n = A.shape
    X = np.zeros((m, n))
    cx = np.int(np.floor((m + 1) / 2))
    cy = np.int(np.floor((n + 1) / 2))
    X[cx, cy] = 1

    for k in range(iters):

        C = nd.binary_dilation(X, B).astype(A.dtype)
        X = np.logical_and(C, complement(A))

        if do_plot:
            plt.subplot(1, 2, 2)
            plt.title("Iteracion: " + str(k))
            plt.imshow(X, cmap='gray')
            plt.pause(0.001)

    return X


A = Image.open("imagen7.jpg", "r").convert("L")
A = np.array(A)
A = to_bin(A)

B = np.ones((4, 4), A.dtype)

plt.figure(1)
plt.subplot(121)
plt.title("Imagen Original")
plt.imshow(A, cmap='gray')

plt.ion()

X = rellenar(A, B, 100, True)

plt.ioff()

plt.subplot(1, 2, 2)
plt.title("Iteracion: 100 [Finalizado]")
plt.imshow(X, cmap='gray', vmin=0, vmax=1, interpolation='none')

plt.show()
