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


class TorchLinearOperator:
    def __init__(self, shape, matvec, matmat, rmatvec=None, rmatmat=None):
        self.shape = shape
        self.matvec, self.matmat = matvec, matmat
        self.rmatvec, self.rmatmat = rmatvec, rmatmat

    def __matmul__(self, other):

        if isinstance(other, TorchLinearOperator):
            return TorchLinearOperator(
                shape=(self.shape[0], other.shape[1]),
                matvec=lambda x: self.matvec(other.matvec(x)),
                matmat=lambda x: self.matmat(other.matmat(x)),
                rmatvec=lambda x: self.rmatvec(other.rmatvec(x)),
                rmatmat=lambda x: self.rmatmat(other.rmatmat(x)),
            )
        if other.ndim == 1:
            return self.matvec(other)

        return self.matmat(other)

    @property
    def T(self):
        return TorchLinearOperator(
            shape=(self.shape[1], self.shape[0]),
            matvec=self.rmatvec, matmat=self.rmatmat,
            rmatvec=self.matvec, rmatmat=self.matmat
        )


def make_blur(type_A, x_shape, size=27, std=8):
    img_size = np.prod(x_shape)
    if type_A == 'denoising':
        A = LinearOperator(
            matvec=lambda x: x,
            matmat=lambda X: X,
            rmatvec=lambda x: x,
            rmatmat=lambda X: X,
            shape=(img_size, img_size),
        )
        A.to_torch = lambda device: A

    elif type_A == 'deblurring':
        filt = np.outer(
            gaussian(size, std),
            gaussian(size, std))
        filt /= filt.sum()
        A = LinearOperator(
            matvec=lambda x: fftconvolve(
                x.reshape(x_shape), filt, mode='same'
            ).flatten(),
            matmat=lambda X: fftconvolve(
                X.reshape(-1, *x_shape), filt, mode='same'
            ).flatten(),
            rmatvec=lambda x: fftconvolve(
                x.reshape(x_shape), filt, mode='same'
            ).flatten(),
            rmatmat=lambda X: fftconvolve(
                X.reshape(-1, *x_shape), filt, mode='same'
            ).flatten(),
            shape=(img_size, img_size),
        )

        def torch_operator(device=None):
            import torch

            blur_operator = torch.nn.Conv2d(
                1, 1, filt.shape, padding="same", bias=False
            )
            blur_operator.weight.data = torch.from_numpy(filt[None, None])
            blur_operator.requires_grad_(False)
            blur_operator = blur_operator.to(device, torch.float32)

            A = TorchLinearOperator(
                matvec=lambda x: blur_operator(
                    x.reshape(1, 1, *x_shape)
                ).flatten(),
                matmat=lambda X: blur_operator(
                    X.reshape(-1, 1, *x_shape)
                ).reshape(X.shape),
                rmatvec=lambda x: blur_operator(
                    x.reshape(1, 1, *x_shape)
                ).flatten(),
                rmatmat=lambda X: blur_operator(
                    X.reshape(-1, *x_shape)
                ).reshape(X.shape),
                shape=(img_size, img_size),
            )

            return A

        A.to_torch = torch_operator
    return A
