import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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


def union(A, B):
    A = to_bin(A)
    B = to_bin(B)
    C = np.logical_or(A, B)
    return C


def intersect(A, B):
    A = to_bin(A)
    B = to_bin(B)
    C = np.logical_and(A, B)
    return C


def diff(A, B):
    A = to_bin(A)
    B = to_bin(B)
    C = A - B
    return to_bin(C)


A = Image.open("imagen2.jpg", "r").convert("L")
B = Image.open("imagen3.jpg", "r").convert("L")
A = np.array(A)
B = np.array(B, dtype=np.double)

plt.figure()
plt.subplot(241)
plt.title("Imagen Binaria A")
plt.imshow(to_bin(A), cmap="gray")

plt.subplot(242)
plt.title("Imagen Binaria B")
plt.imshow(to_bin(B), cmap="gray")

C = complement(A)
plt.subplot(243)
plt.title("Complemento de A")
plt.imshow(C, cmap="gray")

X = complement(B)
plt.subplot(244)
plt.title("Complemento de B")
plt.imshow(X, cmap="gray")

D = union(A, B)
plt.subplot(245)
plt.title("Union A y B")
plt.imshow(D, cmap="gray")

E = intersect(A, B)
plt.subplot(246)
plt.title("Intersección A y B")
plt.imshow(E, cmap="gray")

F = diff(A, B)
plt.subplot(247)
plt.title("Diferencia A y B")
plt.imshow(F, cmap="gray")
plt.show()
