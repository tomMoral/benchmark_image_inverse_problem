from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    import xitorch
    import os
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    load_denoiser = import_ctx.import_from('denoisers', 'load_denoiser')
    import matplotlib.pyplot as plt
    from fft_conv_pytorch import fft_conv, FFTConv1d
    from scipy.sparse.linalg import LinearOperator

    conv = import_ctx.import_from('basicblock', 'conv')
    downsample_avgpool = import_ctx.import_from('basicblock','downsample_avgpool')
    downsample_strideconv = import_ctx.import_from('basicblock','downsample_strideconv')
    sequential = import_ctx.import_from('basicblock', 'sequential')
    ResBlock = import_ctx.import_from('basicblock', 'ResBlock')
    upsample_upconv = import_ctx.import_from('basicblock', 'upsample_upconv')
    upsample_convtranspose = import_ctx.import_from('basicblock', 'upsample_convtranspose')
    single2tensor4 = import_ctx.import_from('basicblock', 'single2tensor4')
    uint2single = import_ctx.import_from('basicblock', 'uint2single')
    UNetRes = import_ctx.import_from('basicblock', 'UNetRes')


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
        

        if self.denoiser_name == 'drunet_gray':
            print(True)
            u_rec = torch.from_numpy(u_rec)
            X_rec = torch.from_numpy(X_rec)
            V_rec = torch.from_numpy(V_rec)
            Y = torch.from_numpy(Y)
            #filt = torch.from_numpy(self.filt)
            x_shape = self.X_shape
            img_size = np.prod(x_shape)
            A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x:    (
                    x.reshape(x_shape), self.filt),
                matmat=lambda X: fft_conv(
                    X.reshape(-1, x_shape), self.filt),
                rmatvec=lambda x: fft_conv(
                    x.reshape(x_shape), self.filt),
                rmatmat=lambda X: fft_conv(
                    X.reshape(-1, x_shape), self.filt),
                shape=(img_size, img_size),
            )
            #A = lambda x: fft_conv(x, self.filt)


            n_channels = 1
            # model_pool = 'model_zoo'
            # model_name = 'drunet_gray' 
            model_path = os.path.join('model_zoo', 'drunet_gray' +'.pth')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.empty_cache()
            model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
            model.load_state_dict(torch.load(model_path), strict=True)
            model.eval()
            for _, v in model.named_parameters():
                v.requires_grad = False
            model = model.to(device)


        print('Start callback')
        while callback(X_rec):
            # Initialization
            Z_j = X_rec
            Z_star = V_rec - u_rec
            plt.figure()
            plt.imshow(np.squeeze(Z_j.reshape(self.X_shape)))
            plt.colorbar()
            plt.title('Initial image')
            plt.show()

            for j in range(self.m1):
                Z_j = Z_j.flatten()
                Z_star = Z_star.flatten()
                #test = self.beta * Z_star
                
                var1 = A.T @ Y
                
                b = var1 + self.beta * Z_star

                var2 = self.A.T @ (self.A @ Z_j) / self.sigma
                if self.denoiser_name == 'drunet_gray':
                    var2 = torch.tensor(var2)
                A_x_est = (
                    var2 + self.beta * Z_j
                )
                res = b - A_x_est
                var3 = self.A.T @ (self.A @ res) / self.sigma
                if self.denoiser_name == 'drunet_gray':
                    var3 = torch.tensor(var3)
                a_res = var3 + self.beta*res
                mu_opt = res.T @ res / (res.T @ a_res)
                Z_j += mu_opt*res
                Z_j[Z_j < 0] = 0
                Z_j[Z_j > 1] = 1

            X_rec = Z_j
            plt.figure()
            plt.imshow(np.squeeze(Z_j.reshape(self.X_shape)))
            plt.colorbar()
            plt.title('Part 1')
            plt.show()

            # relaxation
            x_hat = self.alpha*X_rec.flatten() + (1-self.alpha)*V_rec.flatten()

            # Part 2
            u_rec = u_rec.flatten()
            Z_j = V_rec
            Z_star = X_rec + u_rec

            for j in range(self.m2):
                Z_j = Z_j.reshape(self.X_shape)
                Z_tilt_j = Z_j
                if self.denoiser_name == 'drunet_gray':
                    print('Start denoiser')
                    Z_tilt_j = self.denoiser(Z_j, model, device)
                else:
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
            test = X_rec.numpy()
            
            plt.figure()
            plt.imshow(np.squeeze(test.reshape(self.X_shape)))
            plt.colorbar()
            plt.title('Part 2')
            plt.show()

        print('Finish callback')
        if self.denoiser_name == 'drunet_gray':
            X_rec = X_rec.detach().cpu().numpy()
        print(X_rec.dtype)
        self.X_rec = X_rec.reshape(self.X_shape)


    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.X_rec

