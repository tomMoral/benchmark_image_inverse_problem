# WORK IN PROGRESS


from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')
    load_prox_df = import_ctx.import_from('proximal', 'load_prox_df')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'pnp-hqs'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d', 'nlm'],
        'tau': [0.01, 0.1]
    }

    def set_objective(self, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A = A
        self.Y, self.X_shape = Y.flatten(), X_shape
        self.sigma_f = sigma_f

        self.denoiser = load_denoiser(self.denoiser_name)
        self.prox_f = load_prox_df(self.prox_df)

    def run(self, callback):
        L = get_l2norm(self.A)

        X_rec = np.zeros(self.X_shape)

        while callback(X_rec):
            X_rec = X_rec.flatten()
            u = X_rec - self.tau * self.A.T @ (self.A  @ X_rec - self.Y) / L
            u = u.reshape(self.X_shape)
            X_rec = self.denoiser(image=u, sigma=sqrt(self.tau))

        self.X_rec = X_rec

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
