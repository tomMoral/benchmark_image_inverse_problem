from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import identity

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
        self.maxiter = maxiter
        self.M, self.N = A.shape
        

    def __call__(self, z, alpha):
        """
        compute arg min_x ||y-Ax||^2 + alpha * ||x-z||^2
        """
        # x =  argmin_x ||y-Ax||^2 + alpha * ||x-z|| <=> (AtA + alpha I).x = (At.y + alpha z)
        # assume flatten z if it not flattened
        input_shape = z.shape
        zf = z.flatten()
        Id = identity(self.N, self.N)
        AtA_plus_alphaI = LinearOperator(shape=(self.N, self.N), matvec=lambda x: self.A.T@ self.A + alpha * Id)
        b = self.A.T @ self.y + alpha * zf
        xhat = cg(AtA_plus_alphaI, b = b, tol=self.tol, maxiter=self.maxiter)
        return xhat