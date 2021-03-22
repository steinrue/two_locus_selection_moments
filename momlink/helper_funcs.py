import itertools
import numpy as np
import operator
import scipy.special
import scipy.stats
import scipy.sparse.linalg as ssl
import scipy.linalg
from scipy.sparse import *
import sklearn.neighbors
import math
from scipy.special import binom, gamma

class MomentInterpolator(object):
    def __init__(self, moms, moms_new, interp_type='loglin', renorm=True, clip_ind=True, parsimonious=False, ds_mat=None):
        self.order = moms.order
        self.new_order = moms_new.order
        self.moms = moms
        self.moms_new = moms_new
        self.fmoms = np.array(self.moms.moms, dtype=float)/self.order
        self.fmomsp1 = np.array(self.moms_new.moms, dtype=float)/self.new_order
        self.vtx, self.wts = self.interp_weights()
        self.clip_ind = clip_ind
        self.interp_type = interp_type
        self.renorm = renorm
        self.parsimonious = parsimonious
        self.ds_mat = ds_mat
        if self.interp_type == 'project' or self.parsimonious:
            self.gen_ls_solver()
        if self.interp_type == 'jackknife':
            self.gen_jackknife_matrix()
        if self.interp_type == 'jackknife-constrained':
            self.gen_jackknife_matrix_constrained()
        

    def gen_ls_solver(self):
        # self.ds_mat_fact = ssl.factorized(self.ds_mat)
        pass

    # Function to compute the jacknife. I.e. the approximation of M_{n+1} in terms of M_n
    def interp(self, m):
        if self.clip_ind:
            m = np.clip(m, 0, 1)
        if self.interp_type == 'loglin':
            m = np.clip(m, 0, 1)
            logitm = np.clip(scipy.special.logit(m * ((self.order+1) / (self.order+2))**3), -100, 100)
            logitmp1 = np.einsum('nj,nj->n', np.take(logitm, self.vtx), self.wts)
            out = scipy.special.expit(logitmp1 )* ((self.order+2) /(self.order+1) )**3
        elif self.interp_type == 'lin':
            m = np.clip(m, 0, 1)
            scaled_m = m * ((self.order+1)/ (self.order+2))**3
            mp1 = np.einsum('nj,nj->n', np.take(scaled_m, self.vtx), self.wts)
            out = mp1 * ((self.order+2) /(self.order+1))**3
        elif self.interp_type == 'project':
            out = ssl.lsqr(self.ds_mat, m)[0]
        elif self.interp_type == 'jackknife':
            out= self.jk_mat.dot(m)
        elif self.interp_type =='jackknife-constrained':
            out= self.jk_mat.dot(m)
        if self.clip_ind:
            out = np.clip(out, 0, 1)
        if self.renorm:
            out = out/np.sum(out)
        if self.parsimonious:
            # out = self.ds_mat_fact.solve(m - self.ds_mat.dot(out))
            # import pdb; pdb.set_trace()
            # m = np.clip(m, -10, 10)
            # print(np.max(m))
            out = ssl.lsqr(self.ds_mat, m - self.ds_mat.dot(out))[0] + out

        return out

    # Computes the interpolation vertices and weights
    def interp_weights(self):
        tree = sklearn.neighbors.BallTree(self.fmoms)
        ind = tree.query(self.fmomsp1, k=4, return_distance=False)
        vertices = np.zeros((self.moms_new.nmom, 4), dtype=int)
        weights = np.zeros((self.moms_new.nmom, 4), dtype=float)
        for i, idx in enumerate(ind):
            vertices[i, :] = idx
            X = np.vstack((np.transpose(self.fmoms[idx]), [1, 1, 1, 1]))
            weights[i, :] = np.linalg.lstsq(X, np.append(self.fmomsp1[i], 1), rcond=-1)[0]
        return vertices, weights

    def gen_jackknife_matrix_constrained(self):
        # et 10 nearest neighbors
        tree = sklearn.neighbors.BallTree(self.fmoms)
        nearest = 10
        try_nearest = 20
        ind = tree.query(self.fmomsp1, k=try_nearest, return_distance=False)
        out_mat = lil_matrix((self.moms_new.nmom, self.moms.nmom))
        for i, idx in enumerate(ind):
            closest_idx = []
            x0 = []
            x0_count = 0
            x1 = []
            x1_count = 0
            x2 = []
            x2_count = 0
            for candidate_idx in idx:
                # import pdb; pdb.set_trace()
                candidate_mom = self.moms.moms[candidate_idx]
                add = False
                if (candidate_mom[0] not in x0) and (x0_count < 3):
                    x0.append(candidate_mom[0])
                    add=True
                    x0_count += 1
                if (candidate_mom[1] not in x1) and (x1_count < 3):
                    x1.append(candidate_mom[1])
                    add=True
                    x1_count += 1
                if (candidate_mom[2] not in x2) and (x2_count < 3):
                    x2.append(candidate_mom[2])
                    add=True
                    x2_count += 1
                if add:
                    closest_idx.append(candidate_idx)
                    idx = np.delete(idx, candidate_mom)
                if min([x0_count, x1_count, x2_count]) == 3:
                    extra_needed = nearest - len(closest_idx)
                    closest_idx.extend(idx[:extra_needed])
            if min([x0_count, x1_count, x2_count]) < 3:
                import pdb; pdb.set_trace()
            A = np.zeros((nearest, nearest))
            b = np.zeros(nearest)
            for j in range(nearest):
                a = np.zeros(10)
                a[j] = 1
                b[j] = self.compute_jackknife_moment(self.moms_new.moms[i], a)
                # import pdb; pdb.set_trace()
                for k, mom_idx in enumerate(closest_idx):
                    A[j, k] = self.compute_jackknife_moment(self.moms.moms[mom_idx], a)
            # import pdb; pdb.set_trace()
            coefs = scipy.linalg.lstsq(A, b)[0]
            for ell, mom_idx in enumerate(closest_idx):
                out_mat[i, mom_idx] = coefs[ell]
        self.jk_mat = out_mat.tocsr()

    def gen_jackknife_matrix(self):
        # et 10 nearest neighbors
        tree = sklearn.neighbors.BallTree(self.fmoms)
        nearest = 10
        ind = tree.query(self.fmomsp1, k=nearest, return_distance=False)
        out_mat = lil_matrix((self.moms_new.nmom, self.moms.nmom))
        for i, idx in enumerate(ind):
            A = np.zeros((nearest, nearest))
            b = np.zeros(nearest)
            for j in range(nearest):
                a = np.zeros(10)
                a[j] = 1
                b[j] = self.compute_jackknife_moment(self.moms_new.moms[i], a)
                for k, mom_idx in enumerate(idx):
                    A[j, k] = self.compute_jackknife_moment(self.moms.moms[mom_idx], a)
            # import pdb; pdb.set_trace()
            coefs = scipy.linalg.lstsq(A, b)[0]
            for ell, mom_idx in enumerate(idx):
                out_mat[i, mom_idx] = coefs[ell]
        self.jk_mat = out_mat.tocsr()

    def compute_jackknife_moment(self, mom, a):
        a0 = a[0]
        a1 = a[1]
        a2 = a[2]
        a3 = a[3]
        a4 = a[4]
        a5 = a[5]
        a6 = a[6]
        a7 = a[7]
        a8 = a[8]
        a9 = a[9]
        n1 = mom[0]
        n2 = mom[1]
        n3 = mom[2]
        n4 = mom[3]


        out = (((5*a2 + 5*a3 + 2*a4 + 2*a5 + 2*a6 + a7 + a8 + a9 + a2*n1 + a3*n1 + 
                    3*a4*n1 + a7*n1 + a8*n1 + a4*n1**2 + 6*a2*n2 + a3*n2 + 3*a5*n2 + 
                    a7*n2 + a9*n2 + a2*n1*n2 + a7*n1*n2 + a2*n2**2 + a5*n2**2 + 
                    a2*n3 + 6*a3*n3 + 3*a6*n3 + a8*n3 + a9*n3 + a3*n1*n3 + a8*n1*n3 +
                    a2*n2*n3 + a3*n2*n3 + a9*n2*n3 + a3*n3**2 + 
                    a6*n3**2 + (a2 + a3 + a2*n2 + a3*n3)*n4 + 
                    a1*(1 + n1)*(5 + n1 + n2 + n3 + n4) + 
                    a0*(4 + n1 + n2 + n3 + n4)*(5 + n1 + n2 + n3 + n4))
                    * gamma(1 + n1+ n2 + n3 + n4)) /
                        gamma(6 + n1 + n2 + n3 + n4))
        return out

    

    # Convert initial frequencies into initial moments
    def convert_to_moms(self, freq):
        assert len(freq) == self.moms.hap_num
        freq = np.array(freq)
        freq = freq/np.sum(freq)
        out = np.zeros(self.moms.nmom)
        for i, mom in enumerate(self.moms.moms):
            out[i] = scipy.stats.multinomial.pmf(mom, n=self.order, p=freq)
            assert not math.isnan(out[i])
        assert abs(np.sum(out) - 1) < 1e-8
        return out


class MomentReducer(object):
    def __init__(self, moms, moms_new):
        self.order = moms.order
        self.new_order = moms_new.order
        self.moms = moms
        self.moms_new = moms_new
        self.moms_inbetween = Moments(self.order - self.new_order)

    
    def multinomial(self, params):
        if len(params) == 1:
            return 1
        return binom(sum(params), params[-1]) * self.multinomial(params[:-1])
    
    def computeLower(self,  input_moments):
        out_mom = []
        for new_moment in self.moms_new.moms:
            norm_coef = self.multinomial(new_moment)
            new_freq = 0
            for moment in self.moms_inbetween.moms:
                higher_mom = [i+j for i,j in zip(new_moment, moment)]
                higher_freq = input_moments[self.moms.lookup(tuple(higher_mom))]
                new_freq += norm_coef*self.multinomial(moment)/self.multinomial(higher_mom)*higher_freq
            out_mom.append(new_freq)
        return out_mom



class Moments(object):
    def __init__(self, order, hap_num=4):
        self.order = order
        self.hap_num = hap_num
        self.gen_moms()
        self.nmom = len(self.moms)
        self.gen_lookup()
        
    def gen_moms(self):
        print('Generating order ' + str(self.order) + ' list of moments...')
        self.moms = []
        size = self.hap_num + self.order - 1
        for indices in itertools.combinations(range(size), self.hap_num - 1):
            starts = [0] + [index+1 for index in indices]
            stops = indices + (size,)
            self.moms.append(tuple(map(operator.sub, stops, starts)))

    def gen_lookup(self):
        print('Generating order ' + str(self.order) + ' moment lookup table...')
        self.lookup_table = {moment : i for i, moment in enumerate(self.moms)}

    def lookup(self, mom):
        assert sum(mom) == self.order
        return self.lookup_table[mom]


class MomentMarginalizer(object):
    def __init__(self, momobj):
        self.momobj = momobj

    def marginalizeB(self, freqs):
        out = np.zeros(self.momobj.order+1)
        for i, (mom, freq) in enumerate(zip(self.momobj.moms, freqs)):
            idx = mom[0]+mom[2]
            out[idx] += freq
        return out

    def marginalizeA(self, freqs):
        out = np.zeros(self.momobj.order+1)
        for i, (mom, freq) in enumerate(zip(self.momobj.moms, freqs)):
            idx = mom[0]+mom[1]
            out[idx] += freq
        return out
        
    def condition_on_focal_seg(self, freqs):
        seg_total = 0
        out = freqs
        for i, (mom, freq) in enumerate(zip(self.momobj.moms, freqs)):
            numA = mom[0]+mom[1]
            if numA > 0 and numA < self.momobj.order:
                seg_total += freq
            else:
                out[i] = 0
        out /= seg_total
        return out

    def condition_on_focal_present(self, freqs):
        seg_total = 0
        out = freqs
        for i, (mom, freq) in enumerate(zip(self.momobj.moms, freqs)):
            numA = mom[0]+mom[1]
            if numA > 0:
                seg_total += freq
            else:
                out[i] = 0
        out /= seg_total
        return out

    def condition_on_focal_freq_unfolded(self, freqs, focal_freq):
        seg_total = 0
        out = freqs
        for i, (mom, freq) in enumerate(zip(self.momobj.moms, freqs)):
            numA = mom[0]+mom[1]
            if numA == focal_freq:
                seg_total += freq
            else:
                out[i] = 0
        out /= seg_total
        return out

    def condition_on_focal_freq_folded(self, freqs, focal_freq):
        seg_total = 0
        out = freqs
        for i, (mom, freq) in enumerate(zip(self.momobj.moms, freqs)):
            numA = mom[0]+mom[1]
            if numA == focal_freq or numA == self.momobj.order - focal_freq:
                seg_total += freq
            else:
                out[i] = 0
        out /= seg_total
        return out