from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


def psnr(rec, ref):
    """Compute the peak signal-to-noise ratio for grey images in [0, 1].
    Parameters
    ----------
    rec : numpy.array, shape (height, width)
        reconstructed image
    ref : numpy.array, shape (height, width)
        original image
    Returns
    -------
    psnr : float
        psnr of the reconstructed image
    """
    mse = np.square(rec - ref).mean()
    psnr = 10 * np.log10(1 / mse)

    return mse, psnr


class Objective(BaseObjective):
    name = "Image Inverse Problem"

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return np.zeros(self.X_ref)

    def set_data(self, A, Y, X_ref):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.A, self.Y, self.X_ref = A, Y, X_ref

    def compute(self, X_rec):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        mse, psnr_ = psnr(X_rec, self.X_ref)
        return dict(value=mse, psnr=psnr_)

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(A=self.A, Y=self.Y, X_shape=self.X_ref.shape)
