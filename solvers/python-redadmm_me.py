from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    import matplotlib.pyplot as plt
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'red-admm'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'denoiser_name': ['bm3d'],
        'tau' : [1], 
        'N' : [9],
        'm1': [200],
        'm2':[1], 
        'beta':[0.001],
        'sigma':[0.02],
        'sigma_f':[0.02],
        'lambda_r':[0.002],
        'alpha':[2]
                    }

    def set_objective(self, A, Y, X_shape):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape
        self.denoiser = load_denoiser(self.denoiser_name)

    def run(self, callback):
        L = get_l2norm(self.A)
        Y = (self.Y - np.amin(self.Y)) / (np.amax(self.Y) - np.amin(self.Y))*255
        X_rec = Y.reshape(self.X_shape) #np.zeros(self.X_shape)
        V_rec = Y.reshape(self.X_shape)
        u_rec = np.zeros(self.X_shape)
        
        plt.figure()
        plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
        plt.title("Initial image")
        #plt.colorbar()
        index_k = 1
        iteration = 0 


        while callback(X_rec):
        #for k in range(self.N):
            #print(k)
            # Part 1 
            # Initialization
            Z_j = X_rec
            Z_star = V_rec - u_rec

            for j in range(self.m1):
                Z_j = Z_j.flatten()
                Z_star = Z_star.flatten()
                b= self.A.T @ Y +self.beta *Z_star
                A_x_est = self.A.T @ (self.A @ Z_j) / (L) + self.beta*Z_j
                res = b - A_x_est
                a_res = self.A.T @ (self.A @ res) / (L) + self.beta*res
                mu_opt = res.T @ res /(res.T @ a_res)
                Z_j += mu_opt*res
                # Z_star = Z_star.flatten()
                # e_j = self.A.T @ (self.A  @ Z_j - Y)/(L*self.sigma**2) + self.beta * (Z_j - Z_star)
                # r_j = self.A.T @ (self.A  @ e_j)/(L*self.sigma**2) + self.beta * e_j
                # mu = e_j.T @ e_j /(e_j.T @ r_j)
                # Z_j += mu * e_j
                Z_j[Z_j<0] = 0
                Z_j[Z_j>255] = 255
                # plt.subplot(2,6,j+2)
                # plt.imshow(Z_j.reshape(self.X_shape))
                # plt.colorbar()
                # plt.title(f"{j+1}")
                # plt.show()
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
                Z_tilt_j = self.denoiser(image=Z_j, sigma = self.sigma_f)
                Z_j = Z_j.flatten()
                Z_tilt_j = Z_tilt_j.flatten()
                Z_star = Z_star.flatten()
                Z_j = 1/(self.beta + self.lambda_r) * (self.lambda_r * Z_tilt_j + self.beta * Z_star)
                
                # plt.imshow(Z_j.reshape(self.X_shape))
                # plt.colorbar()
                # plt.title('Second loop')
                # plt.show()
            V_rec = Z_j

            # Part 3
            u_rec  += x_hat - V_rec
            #u_rec += X_rec - V_rec


            iteration += 1 
            # if np.mod(k, 1)==0:
            #     plt.subplot(2,5,index_k+1)
            #     plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
            #     plt.title(f"{str(k)}")
                
            #     index_k = index_k + 1
        plt.show()
        self.X_rec = X_rec.reshape(self.X_shape)
        #self.X_rec = X_rec
        
        plt.figure()

        plt.subplot(121)
        plt.imshow(np.squeeze(self.Y.reshape((self.X_shape))))
        plt.title("Initial image")
        plt.subplot(122)
        plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
        plt.title(f"ADMM, {str(self.denoiser_name)}, {str(iteration)} iterations")
        #plt.title(f"ADMM, no denoiser, {str(iteration)} iterations")
        plt.show()


    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec
