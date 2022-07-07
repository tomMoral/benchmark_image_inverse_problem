from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    import torch

    get_l2norm = import_ctx.import_from("shared", "get_l2norm")
    load_denoiser = import_ctx.import_from("denoisers", "load_denoiser")


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""

    name = "pnp-fb"

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {
        "denoiser_name": ["bm3d", "nlm"],
        "tau": [0.01, 0.1],
        "start": ["zero", "noisy"],
    }

    def skip(self, A, Y, X_shape, sigma_f):
        if self.start == "noisy" and A.shape[0] != A.shape[1]:
            return True, "Noisiy start need square A"
        return False, None

    def set_objective(self, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A = A
        self.Y, self.X_shape = Y.flatten(), X_shape
        self.denoiser = load_denoiser(self.denoiser_name)

    def run(self, callback):
        L = get_l2norm(self.A)

        if self.start == "zero":
            X_rec = np.zeros(self.X_shape)
        elif self.start == "noisy":
            X_rec = self.Y.reshape(self.X_shape)
        else:
            raise ValueError("unknown value for start.")

        Y = self.Y
        A = self.A

        if self.denoiser_name == "drunet_gray":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            X_rec = torch.from_numpy(X_rec).to(device, torch.float32)
            Y = torch.from_numpy(Y).to(device, torch.float32)
            A = self.A.to_torch(device=device)

        while callback(X_rec):
            X_rec = X_rec.flatten()
            u = X_rec - self.tau * (A.T @ (A @ X_rec - Y)) / L
            u = u.reshape(self.X_shape)
            X_rec = self.denoiser(image=u[None], sigma=sqrt(self.tau))

        self.X_rec = X_rec

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
