from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

def load_prox_df(A, y, sigma, maxiter=None, tol=None):
    """
    return a solver to compute arg min_x ||y-Ax||^2 / (2 sigma**2) + alpha * ||x-z||^2 / 2
    """
    return CG_prox_solver(A, y, sigma, maxiter, tol)

class CG_prox_solver():

    def __init__(self, A, y, sigma : float, maxiter: int, tol : float):
        """
        compute arg min_x ||y-Ax||^2 / (2 sigma**2) + alpha * ||x-z||^2 / 2 using scipy.linalg conjugate gradient solver
        Args:
            A (scipy.sparse.linalg.LinearOperator): (take flatten vector)
            y (np.array): _description_
            maxiter (int): max itereration of CG
            tol (float): tolerance of conjugate gradient stopping criterion
        """
        self.A = A
        self.y = y.flatten()
        self.sigma2 = sigma**2
        self.Aty = self.A.T @ self.y
        self.maxiter = maxiter
        self.M, self.N = A.shape
        self.tol = tol
        

    def __call__(self, z, alpha, x0=None):
        """
        compute arg min_x ||y-Ax||^2 / (2 sigma**2) + alpha * ||x-z||^2 / 2
        
        Args:
            z (np.array): input, can flattened or not 
            alpha (_type_): _description_
            x0 (np.array): initialisation of conjugate gradient (optionnal)

        Returns:
            np.array: CG solution (same shape as z)
        """
        # x = arg min_x ||y-Ax||^2 / (2 sigma**2) + alpha * ||x-z||^2 / 2  <=> (AtA + alpha*sigma**2 I).x = (At.y + alpha*sigma**2 z)
        input_shape = z.shape
        zf = z.flatten()
        alpha_s2 = alpha * self.sigma2
        AtA_plus_alpha_s2_I = LinearOperator(shape=(self.N, self.N), matvec=lambda x: self.A.T@ self.A @ x + alpha_s2 * x)
        b = self.Aty + alpha_s2 * zf
        xhat, _ = cg(AtA_plus_alpha_s2_I, b=b, x0=x0.flatten(), tol=self.tol, maxiter=self.maxiter)
        return xhat.reshape(input_shape)