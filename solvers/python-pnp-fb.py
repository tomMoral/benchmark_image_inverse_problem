from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    import matplotlib.pyplot as plt
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'pnp-fb'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d', 'nlm'],
        'tau' : [0.04], 
        'sigma_d':[0.005]
                    }

    def set_objective(self, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape
        self.denoiser = load_denoiser(self.denoiser_name)

    def run(self, callback):
        L = get_l2norm(self.A)

        X_rec = self.Y.reshape(self.X_shape) #np.zeros(self.X_shape)
        iteration = 0 
        while callback(X_rec):
            X_rec = X_rec.flatten()
            u = X_rec -  self.tau * self.A.T @ (self.A  @ X_rec - self.Y) / L
            u = u.reshape(self.X_shape)
            #X_rec = u
            X_rec = self.denoiser(image=u, sigma = self.sigma_d) #np.sqrt(self.tau))
            iteration = iteration + 1
            # if np.mod(iteration,1)==0:
            #     # plt.subplot(211)
            #     # plt.imshow(np.squeeze(self.Y))
            #     # plt.title("Initial image")
            #     #plt.subplot(212)
            #     plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
            #     plt.title(f"Result with denoiser: {str(self.denoiser_name)} after {str(iteration)} iterations")
            #     plt.show()

        self.X_rec = X_rec
        plt.subplot(121)
        plt.imshow(np.squeeze(self.Y.reshape((self.X_shape))))
        plt.title("Initial image")
        plt.subplot(122)
        plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
        #plt.title(f"FB, no denoiser, {str(iteration)} iterations")
        
        plt.title(f"FB, {str(self.denoiser_name)}, {str(iteration)} iterations")
        plt.show()


    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
