from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'red-admm'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d'],
        'tau': [1],
        'N': [50],
        'm1': [200],
        'm2': [1],
        'beta': [0.001],
        'sigma': [1],
        'lambda_r': [0.5],  # [0.002], #0.2 pour 0.002
        'alpha': [2]
    }

    def set_objective(self, filt, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.filt, self.A = filt, A
        self.Y, self.X_shape = Y.flatten(), X_shape
        self.denoiser = load_denoiser(self.denoiser_name)
        self.sigma_f = sigma_f

    def run(self, callback):
        Y = self.Y
        X_rec = Y.reshape(self.X_shape)
        V_rec = Y.reshape(self.X_shape)
        u_rec = np.zeros(self.X_shape)

        while callback(X_rec):
            # Initialization
            Z_j = X_rec
            Z_star = V_rec - u_rec

            for j in range(self.m1):
                Z_j = Z_j.flatten()
                Z_star = Z_star.flatten()
                b = self.A.T @ Y + self.beta * Z_star
                A_x_est = (
                    self.A.T @ (self.A @ Z_j) / self.sigma + self.beta * Z_j
                )
                res = b - A_x_est
                a_res = self.A.T @ (self.A @ res) / self.sigma + self.beta*res
                mu_opt = res.T @ res / (res.T @ a_res)
                Z_j += mu_opt*res
                Z_j[Z_j < 0] = 0
                Z_j[Z_j > 1] = 1

            X_rec = Z_j

            # relaxation
            x_hat = self.alpha*X_rec.flatten() + (1-self.alpha)*V_rec.flatten()
            # Part 2

            u_rec = u_rec.flatten()
            Z_j = V_rec
            Z_star = X_rec + u_rec

            for j in range(self.m2):
                Z_j = Z_j.reshape(self.X_shape)
                Z_tilt_j = Z_j
                Z_tilt_j = self.denoiser(image=Z_j, sigma=self.sigma_f)

                Z_j = Z_j.flatten()
                Z_tilt_j = Z_tilt_j.flatten()
                Z_star = Z_star.flatten()
                Z_j = 1/(self.beta + self.lambda_r) * (
                    self.lambda_r * Z_tilt_j + self.beta * Z_star
                )

            V_rec = Z_j

            # Part 3
            u_rec += x_hat - V_rec

        self.X_rec = X_rec.reshape(self.X_shape)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
