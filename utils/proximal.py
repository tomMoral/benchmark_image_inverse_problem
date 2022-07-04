from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

def load_prox_df(A, y, maxiter=None, tol=None):
    """
    return a solver to compute arg min_x ||y-Ax||^2 + alpha * ||x-z||^2
    """
    return CG_prox_solver(A, y, maxiter, tol)

class CG_prox_solver():

    def __init__(self, A, y, maxiter: int, tol : float):
        """
        compute arg min_x ||y-Ax||^2 + alpha * ||x-z||^2 using scipy.linalg conjugate gradient solver
        Args:
            A (scipy.sparse.linalg.LinearOperator): (take flatten vector)
            y (np.array): _description_
            maxiter (int): max itereration of CG
            tol (float): tolerance of conjugate gradient stopping criterion
        """
        self.A = A
        self.y = y.flatten()
        self.Aty = self.A.T @ self.y
        self.maxiter = maxiter
        self.M, self.N = A.shape
        self.tol = tol
        

    def __call__(self, z, alpha, x0=None):
        """
        compute arg min_x ||y-Ax||^2 + alpha * ||x-z||^2
        
        Args:
            z (np.array): input, can flattened or not 
            alpha (_type_): _description_
            x0 (np.array): initialisation of conjugate gradient (optionnal)

        Returns:
            np.array: CG solution (same shape as z)
        """
        # x =  argmin_x ||y-Ax||^2 + alpha * ||x-z|| <=> (AtA + alpha I).x = (At.y + alpha z)
        input_shape = z.shape
        zf = z.flatten()
        AtA_plus_alphaI = LinearOperator(shape=(self.N, self.N), matvec=lambda x: self.A.T@ self.A @ x + alpha * x)
        b = self.Aty + alpha * zf
        xhat, _ = cg(AtA_plus_alphaI, b=b, x0=x0.flatten(), tol=self.tol, maxiter=self.maxiter)
        return xhat.reshape(input_shape)