from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'red-admm'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d'],
        'N': [50],
        'm1': [200],
        'm2': [1],
        'beta': [0.001],
        'sigma_den': [0.016],
        'lambda_r': [0.002],  # [0.002], #0.2 pour 0.002
        'alpha': [2]
    }

    def set_objective(self, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A = A
        self.Y, self.X_shape = Y, X_shape
        self.denoiser = load_denoiser(self.denoiser_name)
        self.sigma_f = sigma_f
        #self.sigma_f = 1

    def run(self, callback):
        Y, X_rec, V_rec = self.Y.flatten(), self.Y.copy(), self.Y.copy() 
        u_rec = np.zeros(self.X_shape)

        A = self.A
        if self.denoiser_name == 'drunet_gray':

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            u_rec = torch.from_numpy(u_rec).to(device, torch.float32)
            X_rec = torch.from_numpy(X_rec).to(device, torch.float32)
            V_rec = torch.from_numpy(V_rec).to(device, torch.float32)
            Y = torch.from_numpy(Y).to(device, torch.float32)
            A = self.A.to_torch(device=device)

        
        while callback(X_rec):
            # Part 1 - Prox operator for the data fitting term
            X_rec = X_rec.flatten()
            Z_star = (V_rec - u_rec).flatten()
            for _ in range(self.m1):

                b = A.T @ Y + self.beta * Z_star

                A_x_est = (
                    A.T @ (A @ X_rec) / self.sigma_f + self.beta * X_rec
                )
                res = b - A_x_est
                a_res = A.T @ (A @ res) / self.sigma_f + self.beta*res
                mu_opt = res @ res / (res @ a_res)
                X_rec += mu_opt*res
                X_rec[X_rec < 0] = 0
                X_rec[X_rec > 1] = 1

            

            X_rec = X_rec.reshape(self.X_shape)

            # relaxation
            X_hat = self.alpha * X_rec + (1-self.alpha) * V_rec

            # Part 2 - denoising
            Z_star = X_rec + u_rec
            for _ in range(self.m2):
                V_tild = self.denoiser(
                        image=V_rec, sigma=self.sigma_f
                )[0]

                V_rec = 1/(self.beta + self.lambda_r) * (
                    self.lambda_r * V_tild + self.beta * Z_star
                )
            

            # Part 3 - update multiplier
            u_rec += X_hat - V_rec

        self.X_rec = X_rec

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
