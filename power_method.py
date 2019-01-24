import argparse
import cmath
from scipy.linalg import solve
from numpy.linalg import norm
import numpy as np
TOL = 1e-7
EPSILON = np.finfo(float).eps

def is_close(a_1, a_2):
    return cmath.isclose(a_1, a_2, rel_tol=TOL)

def power_method(A):
    eigenvec = np.random.rand(A.shape[0], 1)
    eigenval = eigenvalue(A, eigenvec)
    for _ in range(60000):
        new_eigenvec = A.dot(eigenvec)
        new_eigenvec /= np.linalg.norm(new_eigenvec)
        new_eigenval = eigenvalue(A, new_eigenvec)
        if norm(new_eigenval-eigenval) < EPSILON:
            return new_eigenvec
        eigenvec = new_eigenvec
        eigenval = new_eigenval
    return  eigenvec

def eigenvalue(A, eigvec):
    return np.dot(np.transpose(eigvec), A.dot(eigvec))/np.dot(np.transpose(eigvec), eigvec)

def inverse_power_method(A):
    eigenvec = np.random.rand(A.shape[0], 1)
    eigenval = eigenvalue(A, eigenvec)
    for _ in range(60000):
        new_eigenvec = solve(A, eigenvec)
        new_eigenvec /= np.linalg.norm(new_eigenvec)
        new_eigenval = eigenvalue(A, new_eigenvec)
        if norm(new_eigenval-eigenval) < EPSILON:
            return new_eigenvec
        eigenvec = new_eigenvec
        eigenval = new_eigenval
    return eigenvec

# Householder transformation application
def deflate_matrix(A, eigenvec):
    I = np.asmatrix(np.identity(A.shape[0]))
    e = np.asmatrix(np.zeros(A.shape[0]))
    e = e.T
    e.itemset(0, 1)

    if is_close(eigenvec.item(0), 0):
        beta = 1
    else:
        beta = np.sign(eigenvec.item(0))
    beta *= (-1)
    alfa = (2**0.5)/norm(eigenvec - beta*e)
    vec = alfa*(eigenvec - beta*e)

    H = I - vec.dot(vec.T)

    return H.dot(A).dot(H.T), H

#H - Householder transformation
def determine_eigenvector(H, rows, eigvals, eigval, eigvec):
    for i in range(len(H) - 1, -1, -1):
        first_element = rows[i].dot(eigvec)/(eigval-eigvals[i])
        eigvec = np.asmatrix(np.insert(eigvec.tolist(), 0, first_element)).T
        eigvec = H[i].T.dot(eigvec)

    return eigvec

def multiplied_power_method(A, power_fun, M):
    dim = A.shape[0]
    rang = min(dim, M)
    househld = []
    rows = []
    eigenvec = power_fun(A)
    eigenval = eigenvalue(A, eigenvec)
    eigvals = [eigenval.item(0)]
    eigvecs = eigenvec
    for _ in range(rang - 1):
        A, H = deflate_matrix(A, eigenvec)
        househld.append(H)
        rows.append(A[0, 1:dim])
        A = A[1:dim, 1:dim]
        eigenvec = power_fun(A)
        eigenval = eigenvalue(A, eigenvec)

        eigvecs = np.column_stack((eigvecs, (determine_eigenvector
                                             (househld, rows, eigvals, eigenval, eigenvec))))
        eigvals.append(eigenval.item(0))
    return eigvals, eigvecs

def main():
    parser = argparse.ArgumentParser(description='Power method with inverse version.')
    parser.add_argument('input_fl', help="matrix csv input file")
    parser.add_argument('M', type=int, help="number of eigenvalues and vectors")
    parser.add_argument('output_fl', help="output file with eigenvectors")
    parser.add_argument('inverse', nargs='?', choices={1, -1}, type=int, default=1,
                        help="-1 input matrix is inverse, otherwise 0")
    args = parser.parse_args()

    matrix = np.genfromtxt(args.input_fl, delimiter=",")

    if args.inverse == 1:
        vals, vecs = multiplied_power_method(matrix, power_method, args.M)
    else:
        vals, vecs = multiplied_power_method(matrix, inverse_power_method, args.M)

    np.savetxt(args.output_fl, np.asmatrix(vecs), delimiter=",", fmt="%1.7f")
    print(*np.around(vals, decimals=7))

main()
