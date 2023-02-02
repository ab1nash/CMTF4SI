import tensorly as tl
import numpy as np
import copy
from helpers import *

def cmtf4si(X_, y=None, c_m=None, r=10, omega=None, alpha=1, tol=1e-4, maxiter=800, init='random', printitn=1000):
    """
    CMTF Compute a Coupled Matrix and Tensor Factorization (and recover the Tensor).
    ---------
    :param   'x'  - Tensor
    :param   'y'  - Coupled Matries
    :param  'c_m' - Coupled Modes
    :param   'r'  - Tensor Rank
    :param  'omega'- Index Tensor of Obseved Entries
    :param 'alpha'- Impact factor for HiL part {0.0-1.0}
    :param  'tol' - Tolerance on difference in fit {1.0e-4}
    :param 'maxiters' - Maximum number of iterations {50}
    :param 'init' - Initial guess [{'random'}|'nvecs'|cell array]
    :param 'printitn' - Print fit every n iterations; 0 for no printing {1}
    ---------
    :return
     P: Decompose result.(kensor)
     x: Recovered Tensor.
     V: Projection Matrix.
    ---------
    """
    x = X_.copy()
    # Construct omega if no input
    if omega is None:
        omega = x * 0 + 1
    bool_omeg = np.array(omega, dtype=bool)
    # Extract number of dimensions and norm of x.
    N = len(x.shape)
    normX = np.linalg.norm(x)
    dimorder = np.arange(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4
    maxiters = maxiter

    # Recover or just decomposition
    recover = 0
    if 0 in omega:
        recover = 1

    Uinit = []
    Uinit.append([])
    for n in dimorder[1:]:
        Uinit.append(np.random.random([x.shape[n], r]))
        
    # Set up for iterations - initializing U and the fit.
    # STEP 1: a random V is initialized - y x V =  |mode x rank|
    U = Uinit[:]
    if type(c_m) == int:
        V = np.random.random([y.shape[1], r])
    else:
        V = [np.random.random([y[i].shape[1], r]) for i in range(len(c_m))]
    fit = 0

    # Save hadamard product of each U[n].T*U[n]
    UtU = np.zeros([N, r, r])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = np.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = x * 1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
            # Calculate Unew = X[n]* khatrirao(all U except n, 'r').
            ktr = tl.tenalg.khatri_rao(U, weights=None, skip_matrix=n)
            Unew = np.dot(tl.unfold(x, n) ,ktr)

            # Compute the matrix of coefficients for linear system
            temp = list(range(n))
            temp[len(temp):len(temp)] = list(range(n + 1, N))
            B = np.prod(UtU[temp, :, :], axis=0)
            if int != type(c_m):
                tempCM = [i for i, a in enumerate(c_m) if a == n]
            elif c_m == n:
                tempCM = [0]
            else:
                tempCM = []
            if tempCM != [] and int != type(c_m):
                for i in tempCM:
                    B = B + np.dot(V[i].T, V[i])
                    Unew = Unew + np.dot(y[i], V[i])
                    V[i] = np.dot(y[i].T, Unew)
                    V[i] = V[i].dot(np.linalg.inv(np.dot(Unew.T, Unew)))
            elif tempCM != []:
                B = B + np.dot(V.T, V)
                Unew = Unew + np.dot((alpha)*(y*~bool_omeg)+(y*bool_omeg), V)
                V = np.dot(((alpha)*(y*~bool_omeg)+(y*bool_omeg)).T, Unew)
                V = V.dot(np.linalg.inv(np.dot(Unew.T, Unew)))
            Unew = Unew.dot(np.linalg.inv(B))
            U[n] = Unew
            UtU[n, :, :] = np.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        lamb = np.ones(r)
        final_shape = tuple(u.shape[0] for u in U)
        P = np.dot(lamb.T, tl.tenalg.khatri_rao(U).T)
        P = P.reshape(final_shape)
        x[bool_omeg] = X_[bool_omeg]
        x[~bool_omeg] = P[~bool_omeg]

        fitchange = np.linalg.norm(x - oldX)


        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            if recover == 0:
                print ('CMTF: iterations=',iter, 'f=',fit, 'f-delta=',fitchange)
            else:
                print ('CMTF: iterations=',iter, 'f-delta=',fitchange)
        if flag == 0:
            break

    return P, x, V