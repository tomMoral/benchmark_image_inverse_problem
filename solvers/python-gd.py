from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import matplotlib.pyplot as plt
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'GD'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape

    def run(self, callback):
        L = get_l2norm(self.A)

        X_rec = np.zeros(np.prod(self.X_shape))
        X_rec_acc = np.zeros_like(X_rec)
        X_rec_old = np.zeros_like(X_rec)

        t_new = 1
        plt.imshow(np.squeeze(self.Y.reshape((self.X_shape))))
        plt.title("Y")
        plt.show()
        iteration = 0 
        while callback(X_rec.reshape(self.X_shape)):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                X_rec_old[:] = X_rec  # x in Beck & Teboulle (2009) notation
                X_rec[:] = X_rec_acc  # y in Beck & Teboulle (2009) notation
            X_rec -= self.A.T @ (self.A  @ X_rec - self.Y) / L
            if self.use_acceleration:
                X_rec_acc[:] = (
                    X_rec + (t_old - 1.) / t_new * (X_rec - X_rec_old)
                )
            # plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))

            # plt.title(f"Iteration: {str(iteration)}")
            # plt.show()
            iteration = iteration + 1
        self.X_rec = X_rec.reshape(self.X_shape)

        plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
        plt.title(f"Result with acceleration: {str(self.use_acceleration)} after {str(iteration)} iterations")
        plt.show()


    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
