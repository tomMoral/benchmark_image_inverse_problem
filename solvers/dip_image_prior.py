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

    def set_objective(self, filt, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.filt, self.A, self.Y, self.X_shape = filt, A, Y, X_shape

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(
        self,
        callback,
    ):
        if torch.cuda.device_count() > 0:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        blur_operator = torch.nn.Conv2d(
            1, 1, self.filt.shape, padding="same", bias=False
        )
        blur_operator.weight.data = np_to_torch(self.filt).type(dtype)
        blur_operator.requires_grad_(False)

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
        ).type(dtype)

        mse = torch.nn.MSELoss().type(dtype)

        Y_torch = np_to_torch(self.Y).type(dtype)

        noise_input_saved = self.input_std * torch.rand(
            1, self.input_depth, self.X_shape[0], self.X_shape[1]
        ).type(dtype)

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.learning_rate
            )

        out = net(noise_input_saved)

        while callback(torch_to_np(out)):
            noise_input = (
                noise_input_saved
                + self.reg_noise_std * torch.randn_like(noise_input_saved)
            )
            out = net(noise_input)
            loss = mse(blur_operator(out), Y_torch)

            loss.backward()
            optimizer.step()

        self.X_rec = torch_to_np(out)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
