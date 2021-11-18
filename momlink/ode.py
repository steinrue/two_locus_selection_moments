import numpy as np
import itertools
import operator
import scipy.integrate
import math
import scipy.stats
from scipy.sparse import *
from scipy.special import betainc, beta, binom
import pdb
import scipy.optimize as optimize
import sympy.ntheory.multinomial

from .helper_funcs import MomentInterpolator, Moments
from  .moran import Moran
from .integrator import Integrator

# I use the following ordering of coorindates:
# 0 - AB
# 1 - Ab
# 2 - aB
# 3 - ab
# with A being the allele that is selected for


# Define a class for ode integration

class TwoLocMomOde(object):
    def __init__(self, order, interp_type='loglin', renorm=True, clip_ind=True, parsimonious=False):

        self.order = order
        self.clip_ind = clip_ind
        self.parsimonious = parsimonious
        self.renorm = renorm
        self.interp_type = interp_type
        self.d_size = 64

        # Generate moment objects
        self.moms = Moments(order)
        self.momsp1 = Moments(order+1)
        self.nmom = self.moms.nmom
        self.nmomp1 = self.momsp1.nmom

        # Generate matrices
        self.dr_mat_skel = self.gen_dr_mat().tocsr() # drift
        self.rmut_mats_skel = self.gen_rmut_mat() # mutation
        self.sel_mat_in_skel, self.sel_mat_out_skel = self.gen_sel_mat() # selection
        self.sel_mat_in_skel = self.sel_mat_in_skel.tocsr()
        self.sel_mat_out_skel = self.sel_mat_out_skel.tocsr()
        self.rec_mat_in_skel, self.rec_mat_out_skel = self.gen_rec_mat() # recombination
        self.rec_mat_in_skel = self.rec_mat_in_skel.tocsr()
        self.rec_mat_out_skel = self.rec_mat_out_skel.tocsr()
        self.ds_mat = self.gen_ds_mat().tocsr() # downsampling

        # Generate interpolator
        self.interp = MomentInterpolator(self.moms, self.momsp1, interp_type, renorm, clip_ind, parsimonious, self.ds_mat)

        self.start = None
     
    def set_parameters(self, **params):
        # Set parameters
        self.demog = params['demog']
        self.pop_size = self.demog.n0
        self.lam = params['lam']
        # Code mutations rate entries as follows:
        # 0 -> A -> a
        # 1 -> a -> A
        # 2 -> B -> b
        # 3 -> b -> B
        
        self.sig = 4*self.pop_size*params['s']
        self.rho = 4*self.pop_size*params['r']
        self.init = params['init']
        self.dr_mat = self.lam*self.dr_mat_skel # drift

        # decide on mutation model
        self.theta = [4*self.pop_size*rate for rate in params['mut']]
        if 'mutModel' not in params:
            params['mutModel'] = 'rec'
             
        if (params['mutModel'] == "rec"):
            self.recMut = True
            self.rmut_mat = sum([x * y for (x, y) in zip(self.rmut_mats_skel, self.theta)]).tocsr() # mutation
        elif (params['mutModel'] == "prf"):
            self.recMut = False
            # make sure all PRF
            assert (np.isclose(self.theta[0],0))
            assert (np.isclose(self.theta[1],0))
            assert (np.isclose(self.theta[2],self.theta[3]))
            # no matrices here, cause this is all inhomogeneous
        else:
            # tertium non datur
            assert (False)
        self.sel_mat_in = self.sig*self.sel_mat_in_skel # selection
        self.sel_mat_out = self.sig*self.sel_mat_out_skel
        self.rec_mat_in = self.rho*self.rec_mat_in_skel # recombination
        self.rec_mat_out = self.rho*self.rec_mat_out_skel
        
        # Generate initial conditions
        if self.init == 'Standing':
            try:
                self.moran_model.set_params(self.theta, self.rho)
            except:
                self.moran_model = Moran(self.moms, self.theta, self.rho, self.order)         
            finally:
                self.start = self.moran_model.stationary()
        elif self.init == 'Stationary' and self.recMut:
            if 'initial_freq'in params:
                if params['initial_freq'] is not None:
                    self.start = self.comp_1L_stationary(initial_freq=params['initial_freq'])
                else:
                    self.start = self.comp_1L_stationary()
            else:
                self.start = self.comp_1L_stationary()
        elif self.init == 'StationaryPRF' and not self.recMut:
            if 'initial_freq'in params:
                self.start = self.comp_stationary_prf (initial_freq=params['initial_freq'])
            else:
                raise Exception('Invalid initial condition.')
        elif self.init == 'MarginalMoms':
            if (('initial_freq'in params) and ('marg_mom' in params)):
                self.start = self.comp_stationary_mom_beta(initial_freq=params['initial_freq'], margMom=params['marg_mom'])
            else:
                raise Exception('Invalid initial condition.')
        elif len(self.init) == self.nmom:
            self.start = self.init
        elif len(self.init) == 3 and sum(self.init) <= 1:
            self.init .append(1-sum(self.init ))
            self.start = self.interp.convert_to_moms(self.init)
        elif len(self.init) == 4 and sum(self.init) == 1:
            self.start = self.interp.convert_to_moms(self.init)
        else:
            raise Exception('Invalid initial condition.')

    def set_selection(self, s):
        self.sig = 4*self.pop_size*s
        self.sel_mat_in = self.sig*self.sel_mat_in_skel # selection
        self.sel_mat_out = self.sig*self.sel_mat_out_skel
    
    def set_recombination(self, r):
        self.rho  = 4*self.pop_size*r
        self.rec_mat_in = self.rho*self.rec_mat_in_skel # recombination
        self.rec_mat_out = self.rho*self.rec_mat_out_skel

    def comp_stationary_prf (self, initial_freq=None, polarized=False):
        # this should be alpha = 0, beta = 1 for the stationary density
        # some integrals don't exist though
        if initial_freq is None:
            initial_freq = 1/(2*self.pop_size)
        out = np.zeros(self.moms.nmom)

        assert (np.isclose (self.theta[3], self.theta[2]))
        Theta = self.theta[2]
        Alpha = 0
        Beta = 1
        betaab = 1

        beta_lookup = {}
        for i in range(0, self.order+2):
            for j in range(0, self.order+2):
                # these only work in some cases with Alpha = 0
                # but that's ok, because we should not need the other ones
                if (j > 0):
                    beta_lookup[(j, i, 1)] = beta(Alpha+j, Beta+i)         
                    beta_lookup[(j, i, 1-initial_freq)] = betainc(Alpha+j, Beta+i, 1-initial_freq) * beta(Alpha+j, Beta+i)
                    beta_lookup[(j, i, initial_freq)] = betainc(Alpha+j, Beta+i, initial_freq) * beta(Alpha+j, Beta+i)

        mc = sympy.ntheory.multinomial.multinomial_coefficients(4, int(self.order))
        for i, mom in enumerate(self.moms.moms):
            # Blinked
            if mom[1] == 0:
                for j in range(0, mom[2]+1):
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(j+1, +mom[3], 1)]
                                            -beta_lookup[(j+1, mom[3], initial_freq)])/betaab)
            # blinked (only finite if mom[2] > 0)
            if (mom[0] == 0) and (not polarized) and (mom[2] > 0):
                for j in range(0, mom[3]+1):
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(mom[2], j+1, 1-initial_freq)])/betaab)
            out[i] *= Theta * mc[mom]

        # now we have to distribute the mass that's left on the states with mom[0] == 0 and mom[2] == 0
        totalLeft = 1 - np.sum(out)

        for i, mom in enumerate(self.moms.moms):
            # only the ones where neutral is all ancestral
            if ((mom[0] == 0) and (mom[2] == 0)):
                # mom[1] is number of derived beneficial
                # just the sampling probability for the beneficial
                out[i] += totalLeft * scipy.stats.binom.pmf (mom[1], self.order, initial_freq)

        return out/np.sum(out)

    def comp_stationary_mom_beta (self, initial_freq=None, polarized=False, margMom=None):

        print('Initializing ode')
        if initial_freq is None:
            initial_freq = 1/(2*self.pop_size)
        
        # make sure margMom describe proper marginal moments
        assert (np.min(margMom) >= 0)
        assert (np.isclose(np.sum(margMom),1))
        # they have to be one bigger
        # we are ok with same order
        assert (len(margMom) > 2)

        # downsample the margMom
        n = len(margMom)-1
        downsampled = {}
        downsampled[n] = margMom.copy()
        while (n >= 1):
            n -= 1
            # here we should take the regular one
            upSFS = downsampled[n+1]
            downSFS = np.zeros (n+1)
            # iterate over entries in downSFS
            for i in range(n+1):
                downSFS[i] = (n+1-i)/(n+1) * upSFS[i] + (i+1)/(n+1) * upSFS[i+1]
            # save the regular one
            downsampled[n] = downSFS.copy()
            # if (n < 4):
            assert (np.isclose (np.sum (downSFS), 1))

        # estimate alpha and beta
        m = downsampled[1][1]
        v = downsampled[2][2] - m*m
        # method of moments
        Alpha = (m*(1-m)/v - 1)*m
        Beta = (m*(1-m)/v - 1)*(1-m)

        # prepare container to return
        out = np.zeros(self.moms.nmom)

        # Initialize beta functions
        betaab = beta (Alpha, Beta)
        beta_lookup = {}
        for i in range(0, self.order+2):
            for j in range(0, self.order+2):
                beta_lookup[(j, i, 1)] = beta(Alpha+j, Beta+i)         
                beta_lookup[(j, i, 1-initial_freq)] = betainc(Alpha+j, Beta+i, 1-initial_freq) * beta(Alpha+j, Beta+i)
                beta_lookup[(j, i, initial_freq)] = betainc(Alpha+j, Beta+i, initial_freq) * beta(Alpha+j, Beta+i)

        mc = sympy.ntheory.multinomial.multinomial_coefficients(4, int(self.order))
        for i, mom in enumerate(self.moms.moms):
            # mom[1] is Ab, so this should be Blink
            if mom[1] == 0:
                for j in range(0, mom[2]+1):
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(j, mom[3], 1)]
                                            -beta_lookup[(j, mom[3], 1-initial_freq)])/betaab)
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(j+1, mom[3], 1-initial_freq)]
                                            -beta_lookup[(j+1, mom[3], initial_freq)])/betaab)
            # mom[0] is AB, so this should be blink
            if mom[0] == 0 and not polarized:
                for j in range(0, mom[3]+1):
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(mom[2], j+1, 1-initial_freq)]
                                            -beta_lookup[(mom[2], j+1, initial_freq)])/betaab)
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(mom[2], j, initial_freq)])/betaab)
            out[i] *= mc[mom]

        print (np.sum(out))
        return out/np.sum(out)


    def comp_1L_stationary(self, initial_freq=None):
        if initial_freq is None:
            initial_freq = 1/(2*self.pop_size)
        out = np.zeros(self.moms.nmom)
        # Initialize beta functions
        betaab = beta(self.theta[3], self.theta[2])
        beta_lookup = {}
        for i in range(0, self.order+2):
            for j in range(0, self.order+2):
                # Factors are due to definition of scipy beta, betainc
                beta_lookup[(j, i, 1)] = beta(self.theta[3]+j, self.theta[2]+i)         
                beta_lookup[(j, i, 1-initial_freq)] = betainc(self.theta[3]+j, self.theta[2]+i, 1-initial_freq) * beta(self.theta[3]+j, self.theta[2]+i)
                beta_lookup[(j, i, initial_freq)] = betainc(self.theta[3]+j, self.theta[2]+i, initial_freq) * beta(self.theta[3]+j, self.theta[2]+i)

        mc = sympy.ntheory.multinomial.multinomial_coefficients(4, int(self.order))
        for i, mom in enumerate(self.moms.moms):
            if mom[1] == 0:
                for j in range(0, mom[2]+1):
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(j, mom[3], 1)]
                                            -beta_lookup[(j, mom[3], 1-initial_freq)])/betaab)
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(j+1, mom[3], 1-initial_freq)]
                                            -beta_lookup[(j+1, mom[3], initial_freq)])/betaab)
            if mom[0] == 0:
                for j in range(0, mom[3]+1):
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(mom[2], j+1, 1-initial_freq)]
                                            -beta_lookup[(mom[2], j+1, initial_freq)])/betaab)
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(mom[2], j, initial_freq)])/betaab)
            out[i] *= mc[mom]
        return out/np.sum(out)


    # this one is all inhomogeneous mutations
    def newPRFmut(self, m, Theta):
        # see about all inhomogeneous
        n = self.order
        # make a vector for the net flow
        mutNetFlow = np.zeros(self.moms.nmom)

        #first, what's the total inflow?
        mutInFlux = n * 0.5 * Theta

        # so now, where do they come from, and where do they go
        # where do they come from, cotton eye joe

        # 0 - AB
        # 1 - Ab
        # 2 - aB
        # 3 - ab

        # who can be hit by a mutation?
        # only the b background can be hit
        # so let's go by how many A linked to b
        # 0 is all ancestral (most), n is all beneficial derived
        # first collect all, to see how we divvy up
        flowWeights = np.zeros (n+1)
        for n_A in range(0,n+1):
            srcIdx = self.moms.lookup_table[(0,n_A,0,n-n_A)]
            flowWeights[n_A] = m[srcIdx]
        # and normalize
        flowWeights /= np.sum(flowWeights)

        # and get thereal flow out of every state
        outFlow = flowWeights * mutInFlux

        # and now flow it into the right states            
        for n_A in range(0,n+1):
            srcIdx = self.moms.lookup_table[(0,n_A,0,n-n_A)]
            # in state (0,n_A,0,n-n_A), n_A would flow into (1,n_A-1,0,n-n_A) and n-n_A would flow into (0,n_A,1,n-n_A-1)
            if (n_A == 0):
                dstIdx = self.moms.lookup_table[(0,n_A,1,n-n_A-1)]
                mutNetFlow[srcIdx] -= outFlow[n_A]
                mutNetFlow[dstIdx] += outFlow[n_A]
            elif (n_A == n):
                dstIdx = self.moms.lookup_table[(1,n_A-1,0,n-n_A)]
                mutNetFlow[srcIdx] -= outFlow[n_A]
                mutNetFlow[dstIdx] += outFlow[n_A]
            else:
                # this is the general more complicated one
                derDstIdx = self.moms.lookup_table[(1,n_A-1,0,n-n_A)]
                ancDstIdx = self.moms.lookup_table[(0,n_A,1,n-n_A-1)]
                mutNetFlow[srcIdx] -= outFlow[n_A]
                mutNetFlow[derDstIdx] += outFlow[n_A] * n_A/float(n)
                mutNetFlow[ancDstIdx] += outFlow[n_A] * (n-n_A)/float(n)

        return mutNetFlow



    def comp_stationary_from_B_moms(self, marginal_moms, initial_freq=None):
        '''Takes one-locus moments for neutral site and initial frequency of focal locus and generates two-locus moments
        
        Args:
            marginal_moms: numpy array of sampling probabilities of one-locus biallelec population order should by greater than self.order
            initial_freq: initial frequency of focal locus
        
        Returns:
            Vector of moments of two-locus moments'''

        # Initialize frequency to one allele if not specified
        if initial_freq is None:
            initial_freq = 1/(2*self.pop_size)
        
        # Initialize output, generate multinomial coefficients, and create dictionary of downsampling probabilities
        out = np.zeros(self.moms.nmom)
        mc = sympy.ntheory.multinomial.multinomial_coefficients(4, int(self.order))
        mom_dict = self.comp_marginal_submoments(marginal_moms)

        # Iterate through all moments computing sampling probabilities
        for i, mom in enumerate(self.moms.moms):
            contribution_A = 0
            contribution_a = 0
            if mom[2] == 0:
                contribution_A = mc[mom] / binom(mom[1] + mom[3] + 1, mom[1] + 1) * initial_freq**mom[0] * mom_dict[(mom[1]+mom[3]+1, mom[1]+1)]
            if mom[0] == 0:
                contribution_a = mc[mom] / binom(mom[1] + mom[3] + 1, mom[1]) * initial_freq**mom[2] * mom_dict[(mom[1]+mom[3]+1, mom[1])]
            out[i] = contribution_A + contribution_a
            
        return out

    def comp_marginal_submoments(self, one_locus_moments):
        '''Converts one locus moment vector into dictionary containing all downsampling probabilities

        Args:
            one_locus_moments: numpy array of sampling probabilities of one-locus biallelec population

        Returns:
            Dictionary of single locus moments for all order below the input order'''
        max_order = len(one_locus_moments)-1
        out_moms = {(max_order, i) :  freq for i, freq in enumerate(one_locus_moments)}
        out_moms[(0, 0)] = 1

        for i in range(1, max_order+1):
            for j in range(0, max_order-i+1):
                out_moms[(max_order-i, j)] = (j+1)/(max_order-i+1) * out_moms[(max_order-i+1, j+1)] + (max_order-i+1-j)/(max_order-i+1) * out_moms[(max_order-i+1, j)]
        return out_moms

    # Generate matrix corresponding to evolution of drift
    def gen_dr_mat(self):
        # print('Generating drift matrix...')
        drift_matrix = lil_matrix((self.nmom, self.nmom))
        for i, moment in enumerate(self.moms.moms):
            for j, k in itertools.product(range(4), range(4)):
                if j != k and moment[j] > 0:
                    new_moment = list(moment)
                    new_moment[j] += -1
                    new_moment[k] += 1
                    new_coordinate = self.moms.lookup(tuple(new_moment))
                    drift_matrix[i, new_coordinate] += (.5 * new_moment[j]
                                                        * new_moment[k])
                    drift_matrix[i, i] += -(.5 * moment[j] * moment[k])
        return drift_matrix

    # Generate selection matrix
    def gen_rmut_mat(self):
        # print('Generating recurrent mutation matrix...')

        def theta_hap_comp(j, i, which_loc):
            out = np.zeros(4)
            if which_loc == 0:
                if j==0:
                    out[0] = 1
                else:
                    out[1] = 1
            else:
                if j==0:
                    out[2] = 1
                else:
                    out[3] = 1
            if j==i:
                return -out
            else:
                return out

        kron = np.eye(2)
        rmut_mats = []
        for i in range(4):
            rmut_mats.append(lil_matrix((self.nmom, self.nmom)))
        for i, mom in enumerate(self.moms.moms):
            nu = np.reshape(list(mom), (2, 2))
            rmut_mats[0][i, i] -= nu[0, 0] + nu[0, 1]
            rmut_mats[1][i, i] -= nu[1, 0] + nu[1, 1]
            rmut_mats[2][i, i] -= nu[0, 0] + nu[1, 0]
            rmut_mats[3][i, i] -= nu[0, 1] + nu[1, 1]
            for j, k in itertools.product(range(2), range(2)):
                if nu[j, k] > 0:
                    new_nu = nu.copy()
                    # Frist locus mutation  in
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
        return [.5 * mat for mat in rmut_mats]

    # Generate matrix corresponding to evolution influenced by selection
    def gen_sel_mat(self):
        # print('Generating selection matrix...')
        sel_mat_out = lil_matrix((self.nmom, self.nmomp1))
        sel_mat_in = lil_matrix((self.nmom, self.nmom))
        for i, mom in enumerate(self.moms.moms):
            sel_mat_in[i, i] += 1 / 2 * (mom[0] + mom[1])
            for j in range(2):
                new_mom = list(mom)
                new_mom[j] += 1
                new_coor = self.momsp1.lookup(tuple(new_mom))
                sel_mat_out[i, new_coor] -= (self.order / (self.order + 1)
                                             * 1 / 2 * new_mom[j])
        return sel_mat_in, sel_mat_out

     # Generate matrix corresponding to evolution influenced by recombination
    def gen_rec_mat(self):
        # print('Generating recombination matrix...')
        rec_mat_in = lil_matrix((self.nmom, self.nmomp1))
        rec_mat_out = lil_matrix((self.nmom, self.nmom))
        rec_mat_both = lil_matrix((self.nmom, self.nmomp1))
        for i, mom in enumerate(self.moms.moms):
            rec_mat_out[i, i] -= self.order
            nu = np.reshape(list(mom), (2, 2))
            for j, k in itertools.product(range(2), range(2)):
                new_nu = nu.copy()
                new_nu[j, k] += 1
                new_moment = np.reshape(new_nu, (-1))
                new_coor = self.momsp1.lookup(tuple(new_moment))
                rec_mat_in[i, new_coor] += (nu[j, k]*(nu[j, k]+1)
                                            / (self.order+1))
                new_nu = nu.copy()
                new_nu[1-j, k] += 1
                new_moment = np.reshape(new_nu.copy(), (-1))
                new_coor = self.momsp1.lookup(tuple(new_moment))
                rec_mat_in[i, new_coor] += (nu[j, k]*(nu[1-j, k]+1)
                                            / (self.order+1))
                new_nu = nu.copy()
                new_nu[j, 1-k] += 1
                new_moment = np.reshape(new_nu, (-1))
                new_coor = self.momsp1.lookup(tuple(new_moment))
                rec_mat_in[i, new_coor] += (nu[j, k]*(nu[j, 1-k]+1)
                                            / (self.order+1))
                if nu[j, k] > 0:
                    new_nu = nu.copy()
                    new_nu[j, k] += -1
                    new_nu[1-j, k] += 1
                    new_nu[j, 1-k] += 1
                    new_moment = np.reshape(new_nu, (-1))
                    new_coor = self.momsp1.lookup(tuple(new_moment))
                    rec_mat_in[i, new_coor] += ((nu[1-j, k] + 1)
                                                * (nu[j, 1-k] + 1)
                                                / (self.order + 1))
        return 1/2*rec_mat_in, 1/2*rec_mat_out

    # Generate matrix for downsampling from moments of order n+1 to n
    def gen_ds_mat(self):
        # print('Generating downsampling matrix...')
        ds_mat = lil_matrix((self.nmom, self.nmomp1))
        for i, mom in enumerate(self.moms.moms):
            for j in range(4):
                new_mom = list(mom)
                new_mom[j] += 1
                new_coor = self.momsp1.lookup(tuple(new_mom))
                ds_mat[i, new_coor] += new_mom[j] / (self.order + 1)
        return ds_mat

    # Compute derivative of moments
    def dmdt(self, t, m, restart_thresh):
        gen = int(np.round(2*self.pop_size*t, 0))
        current_pop_size = self.demog.get_popsize_at(gen)
        mp1 = self.interp.interp(m)

        # Downsample
        m_est = self.ds_mat.dot(mp1)

        # Drift
        dr = self.pop_size/current_pop_size  * self.dr_mat.dot(m)

        # Mutation
        if self.recMut:
            # recurrent
            mut = self.rmut_mat.dot(m)
        else:
            # PRF
            mut = self.newPRFmut (m, self.theta[3])

        # Selection
        sel_in = self.sel_mat_in.dot(m_est)
        sel_out = self.sel_mat_out.dot(mp1)

        # Recombination
        rec_out = self.rec_mat_out.dot(m_est)
        rec_in = self.rec_mat_in.dot(mp1)
        rec = rec_in+rec_out

        # Collect all terms
        out = dr + sel_in + sel_out + rec + mut
        
        self.dmdt_evals += 1
        return out


    def integrate_forward_RK45(self, gens, first_gen=0, smallest_step=10**-8, keep_traj = False, time_steps = None, num_points=1, min_step_size=10**-4, d_size = 0):
        if d_size == 32:
            self.d_size = np.float32
        elif d_size == 128:
            self.d_size = np.float128
        else:
            self.d_size = np.float64
        T = gens/(2*self.pop_size)
        t0 = first_gen/(2*self.pop_size)
        intobj = Integrator(lambda t, x: self.dmdt(t, x, restart_thresh=np.inf), clip=self.clip_ind, renorm=self.renorm)
        self.start = self.start.astype(self.d_size)
        self.ds_mat = self.ds_mat.astype(self.d_size)
        self.dr_mat = self.dr_mat.astype(self.d_size)
        if self.recMut:
            self.rmut_mat = self.rmut_mat.astype(self.d_size)
        self.sel_mat_in = self.sel_mat_in.astype(self.d_size)
        self.sel_mat_out = self.sel_mat_out.astype(self.d_size)
        self.rec_mat_out = self.rec_mat_out.astype(self.d_size)
        self.rec_mat_in = self.rec_mat_in.astype(self.d_size)

        self.dmdt_evals = 0


        if time_steps:
            time_steps = [time/(2*self.pop_size) for time in time_steps]
        if keep_traj:
            sol = intobj.solve_ivp_rk45(y0=self.start, t0=t0, T=T, time_points=time_steps, num_points=num_points, min_step_size=min_step_size)
            return [sol[0], sol[1]]

        else:
            sol = intobj.solve_ivp_rk45(y0=self.start, T=T, min_step_size=min_step_size)
            return sol[1][-1]
        
class StepTooLarge(Exception):
    """" Raised when the ODE becomes too unstable"""
    pass

class ODEunstable(Exception):
    """ Raised when ODE become too unstable and cannot decrease step size"""
    pass
