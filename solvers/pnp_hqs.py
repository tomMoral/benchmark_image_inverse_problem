# WORK IN PROGRESS


from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')
    load_prox_df = import_ctx.import_from('proximal', 'load_prox_df')


class Solver(BaseSolver):
    """PnP half quadratic splitting, as proposed in https://arxiv.org/pdf/2008.13751.pdf (eqs 6.a, 6.b)"""
    name = 'pnp-hqs'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d', 'nlm'],
        'tau' : [0.01, 0.1],
        'lambda_r' : [0.23]
                    }

    def set_objective(self, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape
        # TODO : get self.sigma from objective
        self.sigma = 0.1
        self.denoiser = load_denoiser(self.denoiser_name)
        # TODO : add tolerance and maxiter as hyperparameters?
        self.prox_f = load_prox_df(self.A, self.Y, maxiter=100, tol=0.0001)

    def run(self, callback):
        L = get_l2norm(self.A)

        X_k = np.zeros(self.X_shape)
        Z_k = self.Y.copy() # this will not work for all inverse problem

        while callback(X_k):
            mu_k = 1 # TODO : choice of mu ?
            alpha_k = mu_k*self.sigma**2
            X_k = self.prox_f(Z_k, alpha=alpha_k)
            sigma_k = sqrt(self.lambda_r / mu_k)
            Z_k = self.denoiser(image=X_k, sigma = sigma_k) 

        self.X_rec = Z_k 

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
