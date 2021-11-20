import numpy as np
import scipy.sparse as sp
import scipy.special
import itertools
from scipy.linalg import norm
import pdb

class Moran(object):
    def __init__(self, moments, theta, rho, order):
        self.moms = moments
        self.nmom = self.moms.nmom
        self.order = order
        self.construct_copy_mat()
        self.construct_mut_mat()      
        self.construct_rec_mat()
        self.set_params(theta, rho)

    def set_params(self, theta, rho):
        self.theta = theta
        self.rho = rho
        self.mut_mat = sum([.5 * x * y for (x, y) in zip(self.mut_mats_skel, self.theta)]).tocsr()
        self.rec_mat = self.rho/2 * self.rec_mat_skel / (self.order-1)


    def construct_copy_mat(self):
        print("Generating moran copy matrix...")
        copy_matrix = sp.lil_matrix((self.nmom, self.nmom))
        for i, moment in enumerate(self.moms.moms):
            for j, k in itertools.product(range(4), range(4)):
                if j != k and moment[j] > 0:
                    new_moment = list(moment)
                    new_moment[j] += -1
                    new_moment[k] += 1
                    new_coordinate = self.moms.lookup(tuple(new_moment))
                    copy_matrix[new_coordinate, i] += (moment[j]
                                                        * moment[k])
                    copy_matrix[i, i] += -(moment[j] * moment[k])
        self.copy_matrix = .5*copy_matrix.tocsr()
    
    def construct_mut_mat(self):
        print("Generating moran mutation matrix...")
        rmut_mats = []
        for i in range(4):
            rmut_mats.append(sp.lil_matrix((self.nmom, self.nmom)))
        for i, mom in enumerate(self.moms.moms):
            nu = np.reshape(list(mom), (2, 2))
            for j, k in itertools.product(range(2), range(2)):
                # First locus mutations in
                if nu[j, k] > 0:
                    new_nu = nu.copy()
                    new_nu[1-j, k] += 1
                    new_nu[j, k] += -1
                    new_coor = self.moms.lookup(tuple(np.reshape(new_nu, (-1))))
                    if j == 1:
                        rmut_mats[1][new_coor, i] += nu[j, k]
                    else:
                        rmut_mats[0][new_coor, i] += nu[j, k]

                    # Second locus mutations in
                    new_nu = nu.copy()
                    new_nu[j, 1-k] += 1
                    new_nu[j, k] += -1
                    new_coor = self.moms.lookup(tuple(np.reshape(new_nu, (-1))))
                    if k == 1:
                        rmut_mats[3][new_coor, i] += nu[j, k]
                    else:
                        rmut_mats[2][new_coor, i] += nu[j, k]

                # Both loci mutations out
                if j == 1:
                    rmut_mats[1][i, i] -= nu[j, k]
                else:
                    rmut_mats[0][i, i] -= nu[j, k]
                if k == 1:
                    rmut_mats[3][i, i] -= nu[j, k]
                else:
                    rmut_mats[2][i, i] -= nu[j, k]

        self.mut_mats_skel = rmut_mats


    def construct_rec_mat(self):
        print("Generating moran recomination matrix...")
        rec_mat= sp.lil_matrix((self.nmom, self.nmom))
        for i, mom in enumerate(self.moms.moms):
            nu = np.reshape(list(mom), (2, 2))
            for j1, j2, k1, k2 in itertools.product(range(2), range(2), range(2), range(2)):
                if j1 != k1 and j2 != k2 and min(nu[j1, j2], nu[k1, k2]) > 0:
                    new_mom = nu.copy()
                    new_mom[j1, j2] -= 1
                    new_mom[k1, k2] -= 1
                    new_mom[j1, k2] += 1
                    new_mom[j2, k1] += 1
                    new_coor = self.moms.lookup(tuple(np.reshape(new_mom, (-1))))
                    rec_mat[i, i] -= nu[j1, j2] * nu[k1, k2]
                    rec_mat[new_coor, i] += nu[j1, j2] * nu[k1, k2]
        self.rec_mat_skel = rec_mat.tocsr()


    def stationary(self, init=None, norm_order=1):
        print("Iterating moran matrix to stationarity...")
        if min(self.theta) > 0:
            Q = self.copy_matrix + self.mut_mat + self.rec_mat
            size = Q.shape[0]
            assert size == Q.shape[1]
            l = Q.min() * 1.001
            P = sp.eye(size, size) - Q/l
            P =  P.tocsr()        
            pi = np.real(sp.linalg.eigs(P, k=1)[1])
            pi = np.reshape(pi, -1)
            pi /= np.sum(pi)
            # pdb.set_trace()
            assert(abs(sum(abs(pi))-1) <.001)
            return pi
        else:
            raise Exception('Must have positive mutation rates to comptue stationary distribution.')

