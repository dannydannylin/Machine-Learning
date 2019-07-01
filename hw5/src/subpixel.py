import numpy as np

def zero_interpolation(A):
    length = A.shape[0] + A.shape[0] - 1
    new_A = np.zeros((length, length), dtype = np.float32)
    i_A = j_A = 0
    for i in range(length):
        for j in range(length):
            if (i + 1) % 2 == 1 and (j + 1) % 2 == 1:
                new_A[i, j] = A[i_A, j_A]
                j_A += 1
                if j_A % A.shape[0] == 0:
                    j_A = 0
                    i_A += 1
    return new_A

def conv(A, K):
    length = A.shape[0] - K.shape[0] + 1
    kernel_size = K.shape[0]
    F = np.zeros((length, length), dtype = np.float32)
    for i in range(length):
        for j in range(length):
            F[i, j] = np.sum(np.multiply(A[i:i + kernel_size, j: j + kernel_size],K))
    return F

def subpixel_convolution(A, K):
    A = zero_interpolation(A)
    # print(A)
    pad_size = K.shape[0] - 1
    A = np.pad(A, (pad_size, pad_size), 'constant', constant_values = 0)
    # print(A)
    A = conv(A, K)
    return A

def ReLU(A):
    return np.abs(np.multiply(A, (A >= 0.).astype(np.float32)))

A = np.array([
[-1, -2],
[2, 1],
], dtype = np.float32)
K = np.array([
[0, 1, 1, 0],
[-2, 0, -2, 0],
[0, 1, 0, 1],
[1, 1, 1, 1]
], dtype = np.float32)

A = subpixel_convolution(A, K)
A = ReLU(A)
print(A)

