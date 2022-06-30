from pickletools import optimize
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import ipdb
    import numpy as np
    import torch

    get_net = import_ctx.import_from("models", "get_net")
    np_to_torch = import_ctx.import_from("models", "np_to_torch")
    torch_to_np = import_ctx.import_from("models", "torch_to_np")
    get_l2norm = import_ctx.import_from("shared", "get_l2norm")


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""

    name = "DIP"

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape

    def run(self, callback):
        dtype = torch.FloatTensor
        pad = "zero"
        var = 1.0 / 10
        lr = 1e-3
        # max_it = 1000
        reg_noise_std = 1.0 / 30

        # print(self.Y.shape)

        net = get_net(
            1,
            "skip",
            pad,
            n_channels=1,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode="bilinear",
        ).type(dtype)

        mse = torch.nn.MSELoss().type(dtype)

        Y_torch = np_to_torch(self.Y.reshape(self.X_shape)).type(dtype)

        noise_input_saved = var * torch.rand_like(Y_torch)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # ipdb.set_trace()
        out = net(noise_input_saved)
        # print("Before while")
        while callback(torch_to_np(out)):
            noise_input = noise_input_saved + reg_noise_std * torch.randn_like(
                noise_input_saved
            )
            out = net(noise_input)
            loss = mse(out, Y_torch)

            loss.backward()
            optimizer.step()
            # print("Inside while")

        self.X_rec = torch_to_np(out)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
