# WORK IN PROGRESS


from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')
    load_prox_df = import_ctx.import_from('proximal', 'load_prox_df')


class Solver(BaseSolver):
    """PnP half quadratic splitting, as proposed in https://arxiv.org/pdf/2008.13751.pdf (see eqs 6.a, 6.b)"""
    name = 'pnp-hqs'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d', 'nlm'],
        'lambda_r' : [0.23],
        'Kmax' : [50]
                    }

    def set_objective(self, filt, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y, X_shape
        self.sigma_f = sigma_f
        self.denoiser = load_denoiser(self.denoiser_name)
        # TODO : add tolerance and maxiter as hyperparameters?
        self.prox_f = load_prox_df(self.A, self.Y, self.sigma_f, maxiter=100, tol=0.0001)
        self.sigmas_k = np.linspace(49/255, sigma_f, self.Kmax)

    def run(self, callback):

        X_k = self.Y.copy()
        Z_k = self.Y.copy() # this will not work for SR
        i = 0
        while callback(X_k):
            sigma_k = self.sigmas_k[i] if i < self.Kmax else self.sigma_f
            alpha_k = self.lambda_r / sigma_k**2
            X_k = self.prox_f(Z_k, alpha=alpha_k, x0=X_k) # = arg min_x ||Ax-y||**2/(2*sigma**2) + alpha_k / 2 ||x-z_k||**2
            Z_k = self.denoiser(image=X_k, sigma = sigma_k) 
            i += 1
        self.X_rec = Z_k 

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
