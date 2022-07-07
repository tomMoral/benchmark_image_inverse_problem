# WORK IN PROGRESS


from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')
    load_prox_df = import_ctx.import_from('proximal', 'load_prox_df')


class Solver(BaseSolver):
    """PnP half quadratic splitting, as proposed in
    https://arxiv.org/pdf/2008.13751.pdf (see eqs 6.a, 6.b)

    """
    name = 'pnp-hqs'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d', 'nlm'],
        'lambda_r': [0.5],
        'Kmax': [50]
    }

    def set_objective(self, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y, X_shape
        self.sigma_f = sigma_f
        self.denoiser = load_denoiser(self.denoiser_name)
        # TODO : add tolerance and maxiter as hyperparameters?
        A = self.A
        self.sigmas_k = np.linspace(49/255, sigma_f, self.Kmax)
        if self.denoiser_name == 'drunet_gray':

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            Y = torch.from_numpy(Y).to(device, torch.float32)
            A = self.A.to_torch(device=device)

        self.prox_f = load_prox_df(
            A, self.Y, self.sigma_f, maxiter=100, tol=0.0001
        )


    def run(self, callback):

        X_k = self.Y.copy()
        Z_k = self.Y.copy()  # this will not work for SR
        i = 0
        while callback(X_k):
            sigma_k = self.sigmas_k[i] if i < self.Kmax else self.sigma_f
            alpha_k = sigma_k**2 / self.lambda_r

            # arg min_x alpha * ||Ax-y||**2/(2*sigma**2) +  ||x-z_k||**2 / 2
            X_k = self.prox_f(Z_k, alpha=alpha_k, x0=X_k)

            Z_k = self.denoiser(image=X_k, sigma=sigma_k)
            i += 1
        self.X_rec = Z_k

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
