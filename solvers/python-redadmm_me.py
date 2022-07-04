from benchopt import BaseSolver, safe_import_context
from math import sqrt

with safe_import_context() as import_ctx:
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage
    from skimage.measure import compare_ssim
    #from skimage.metrics import structural_similarity as ssim
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
        'N' : [50],
        'm1': [200],
        'm2':[1], 
        'beta':[0.001],
        'sigma':[1],
        #'sigma_f':[0.02],
        'lambda_r':[0.5],#[0.002], #0.2 pour 0.002
        'alpha':[2]
                    }

    def set_objective(self, A, Y, X_shape,sigma_f):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.A, self.Y, self.X_shape = A, Y.flatten(), X_shape
        self.denoiser = load_denoiser(self.denoiser_name)
        self.sigma_f = sigma_f

    def run(self, callback):
        L = get_l2norm(self.A)
        #print('L='+str(L))
        Y = self.Y #(self.Y - np.amin(self.Y)) / (np.amax(self.Y) - np.amin(self.Y))*255
        X_rec = Y.reshape(self.X_shape) #np.zeros(self.X_shape)
        V_rec = Y.reshape(self.X_shape)
        u_rec = np.zeros(self.X_shape)
        
        plt.figure()
        plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
        plt.title("Initial image")
        plt.show()
        #plt.colorbar()
        index_k = 1
        iteration = 0 


        #while callback(X_rec):
        for k in range(self.N):
            #print(k)
            # Part 1 
            # Initialization
            Z_j = X_rec
            Z_star = V_rec - u_rec

            for j in range(self.m1):
                Z_j = Z_j.flatten()
                Z_star = Z_star.flatten()
                b= self.A.T @ Y +self.beta *Z_star
                #A_x_est = self.A.T @ (self.A @ Z_j) / (L) + self.beta*Z_j
                A_x_est = self.A.T @ (self.A @ Z_j) /self.sigma + self.beta*Z_j
                res = b - A_x_est
                #a_res = self.A.T @ (self.A @ res) / (L) + self.beta*res
                a_res = self.A.T @ (self.A @ res) /self.sigma + self.beta*res
                mu_opt = res.T @ res /(res.T @ a_res)
                Z_j += mu_opt*res
                Z_j[Z_j<0] = 0
                Z_j[Z_j>1] = 1
                
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
            
            V_rec = Z_j

            # Part 3
            u_rec  += x_hat - V_rec

            data_fidelity = (self.A @ X_rec -  self.Y).T @ (self.A @ X_rec -  self.Y)
            second_term = X_rec.T @ (X_rec - self.denoiser(X_rec.reshape(self.X_shape),sigma = self.sigma_f).flatten())
            print('data_fidelity='+str(data_fidelity))
            #ssim = compare_ssim(X_rec,  self.Y, full=True)
            #ssim = ssim(X_rec , self.Y, data_range=X_rec.max() - X_rec.min())
            #print('ssim'+str(ssim))
            print('second_term='+str(second_term))


            iteration += 1 
            if np.mod(iteration,10)==0:
                plt.subplot(121)
                plt.imshow(np.squeeze(self.Y.reshape((self.X_shape))))
                plt.title("Initial image")
                plt.subplot(122)
                plt.imshow(np.squeeze(X_rec.reshape((self.X_shape))))
                plt.title(f"ADMM, {str(self.denoiser_name)}, {str(iteration)} iterations")
                plt.show()

        

        self.X_rec = X_rec.reshape(self.X_shape)#/255

        
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
