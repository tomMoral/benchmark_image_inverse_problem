from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import torch

    get_net = import_ctx.import_from("models", "get_net")
    np_to_torch = import_ctx.import_from("models", "np_to_torch")
    torch_to_np = import_ctx.import_from("models", "torch_to_np")
    get_l2norm = import_ctx.import_from("shared", "get_l2norm")


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""

    name = "DIP"

    stopping_criterion = SufficientProgressCriterion(
        patience=50, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        "learning_rate": [0.1],
        "optimizer": ["SGD"],
        "input_depth": [32],
        "pad": ["reflection"],
        "input_std": [0.1],
        "reg_noise_std": [0.03],
        "net_type": ["skip"],
    }

    def set_objective(self, A, Y, X_shape, sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape, self._sigma_f = A, Y, X_shape, sigma_f

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(
        self,
        callback,
    ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        A = self.A.to_torch(device=device)

        net = get_net(
            input_depth=self.input_depth,
            net_type=self.net_type,
            pad=self.pad,
            n_channels=1,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode="bilinear",
        ).to(device=device)

        mse = torch.nn.MSELoss()

        Y_torch = np_to_torch(self.Y).to(device, torch.float32)

        noise_input_saved = self.input_std * torch.rand(
            1, self.input_depth, self.X_shape[0], self.X_shape[1]
        ).to(device, torch.float32)

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.learning_rate
            )
        else:
            raise ValueError(self.optimizer)

        out = net(noise_input_saved)

        while callback(torch_to_np(out)):
            noise_input = (
                noise_input_saved
                + self.reg_noise_std * torch.randn_like(noise_input_saved)
            )
            out = net(noise_input)
            loss = mse(A @ out, Y_torch)

            loss.backward()
            optimizer.step()

        self.X_rec = torch_to_np(out)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
