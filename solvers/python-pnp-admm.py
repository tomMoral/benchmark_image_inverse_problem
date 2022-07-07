# WORK IN PROGRESS


from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')
    load_prox_df = import_ctx.import_from('proximal', 'load_prox_df')


class Solver(BaseSolver):
    """PnP ADMM, as proposed in https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/GlobalSIP2013a.pdf (we use the notation of ruy2019)"""
    name = 'pnp-admm'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d', 'nlm'],
        'alpha' : [0.1],
        'sigma_den' : [0.2] 
                    }

    def set_objective(self, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y, X_shape
        self.sigma_f = sigma_f
        self.denoiser = load_denoiser(self.denoiser_name)
        Y = self.Y
        A = self.A
        # TODO : add tolerance and maxiter as hyperparameters?
        if self.denoiser_name == 'drunet_gray':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            Y = torch.from_numpy(Y).to(device, torch.float32)
            A = self.A.to_torch(device=device)

        self.prox_f = load_prox_df(A, Y, self.sigma_f, maxiter=100, tol=0.0001)
        self.sigma_f = sigma_f

    def run(self, callback):
        # TODO : general initialisation (this will not work for SR)
        X_k = Y.copy()
        Y_k = Y.copy()  
        U_k = Y.copy() * 0
        while callback(X_k):
            # we use ruy2019 notation 
            X_k = self.denoiser(Y_k-U_k, sigma=self.sigma_den) 
            Y_k = self.prox_f(X_k+U_k, alpha=self.alpha) 
            U_k = U_k + X_k - Y_k
        self.X_rec = X_k

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
