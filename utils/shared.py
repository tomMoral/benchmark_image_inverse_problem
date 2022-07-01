import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
from scipy.sparse.linalg import LinearOperator


def get_l2norm(A, n_iter=100):
    # multiplication for the smaller size of matrice
    if A.shape[0] < A.shape[1]:
        A = A.T
    AtA = A.T @ A
    x = np.random.randn(A.shape[1])
    for _ in range(n_iter):
        x = AtA @ x
        x /= np.linalg.norm(x)
    return np.sqrt(np.linalg.norm(AtA @ x))


def make_blur(type_A, x_shape, size=27, std=8):
    img_size = np.prod(x_shape)
    if type_A == 'denoising':
        A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: x,
                matmat=lambda X: X,
                rmatvec=lambda x: x,
                rmatmat=lambda X: X,
                shape=(img_size, img_size),
            )
        filt = np.zeros((size, size))
        filt[size//2, size//2] = 1
    elif type_A == 'deblurring':
        filt = np.outer(
            gaussian(size, std),
            gaussian(size, std))
        filt /= filt.sum()
        A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: fftconvolve(
                    x.reshape(x_shape), filt, mode='same'
                ).flatten(),
                matmat=lambda X: fftconvolve(
                    X.reshape(-1, x_shape), filt, mode='same'
                ).flatten(),
                rmatvec=lambda x: fftconvolve(
                    x.reshape(x_shape), filt, mode='same'
                ).flatten(),
                rmatmat=lambda X: fftconvolve(
                    X.reshape(-1, x_shape), filt, mode='same'
                ).flatten(),
                shape=(img_size, img_size),
            )
    return filt, A
