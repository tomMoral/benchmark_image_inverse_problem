from benchopt import BaseSolver, safe_import_context
import bm3d

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'pnp-fb'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape

    def run(self, callback):
        L = get_l2norm(self.A)

        X_rec = np.zeros(self.X_shape)

        while callback(X_rec):
            # TODO : choice of t
            t = 0.1
            X_rec = X_rec.flatten()
            u = X_rec -  t * self.A.T @ (self.A  @ X_rec - self.Y) / L
            u = u.reshape(self.X_shape)
            # TODO : choice of sigma_psd
            X_rec = bm3d.bm3d(u, sigma_psd = 0.1)

        self.X_rec = X_rec

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
