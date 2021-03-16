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
        self.theta = [4*self.pop_size*rate for rate in params['mut']]
        self.sig = 4*self.pop_size*params['s']
        self.rho = 4*self.pop_size*params['r']
        self.init = params['init']
        self.dr_mat = self.lam*self.dr_mat_skel # drift
        self.rmut_mat = sum([x * y for (x, y) in zip(self.rmut_mats_skel, self.theta)]).tocsr() # mutation 
        self.sel_mat_in = self.sig*self.sel_mat_in_skel # selection
        self.sel_mat_out = self.sig*self.sel_mat_out_skel
        self.rec_mat_in = self.rho*self.rec_mat_in_skel # recombination
        self.rec_mat_out = self.rho*self.rec_mat_out_skel
        
        # Generate initial conditions
        if self.init == 'Stationary':
            if 'initial_freq'in params:
                if params['initial_freq'] is not None:
                    self.start = self.comp_1L_stationary(initial_freq=params['initial_freq'])
                else:
                    self.start = self.comp_1L_stationary()
            else:
                self.start = self.comp_1L_stationary()
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

    def comp_1L_stationary(self, initial_freq=None, polarized=False):
        if initial_freq is None:
            initial_freq = 1/(2*self.pop_size)
        out = np.zeros(self.moms.nmom)
        # Initialize beta functions
        betaab = beta(self.theta[3], self.theta[2])
        beta_lookup = {}
        for i in range(0, self.order+1):
            for j in range(0, self.order+1):
                beta_lookup[(self.theta[3]+j+1, self.theta[2]+i, 1)] = beta(self.theta[3]+j+1, self.theta[2]+i)         
                beta_lookup[(self.theta[3]+j+1, self.theta[2]+i, 1-initial_freq)] = betainc(self.theta[3]+j+1, self.theta[2]+i, 1-initial_freq) * beta(self.theta[3]+j+1, self.theta[2]+i)
                beta_lookup[(self.theta[3]+j+1, self.theta[2]+i, initial_freq)] = betainc(self.theta[3]+j+1, self.theta[2]+i, initial_freq) * beta(self.theta[3]+j+1, self.theta[2]+i)
                beta_lookup[(self.theta[3]+i, self.theta[2]+j+1, 1-initial_freq)] = betainc(self.theta[3]+i, self.theta[2]+j+1, 1-initial_freq) * beta(self.theta[3]+i, self.theta[2]+j+1)
                beta_lookup[(self.theta[3]+i, self.theta[2]+j+1, initial_freq)] = betainc(self.theta[3]+i, self.theta[2]+j+1, initial_freq) * beta(self.theta[3]+i, self.theta[2]+j+1)

        mc = sympy.ntheory.multinomial.multinomial_coefficients(4, int(self.order))
        for i, mom in enumerate(self.moms.moms):
            if mom[1] == 0:
                for j in range(0, mom[2]+1):
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(self.theta[3]+j+1, self.theta[2]+mom[3], 1)]
                                            -beta_lookup[(self.theta[3]+j+1, self.theta[2]+mom[3], 1-initial_freq)])/betaab)
                    out[i] += (binom(mom[2], j) * (-initial_freq)**(mom[2]-j) * (initial_freq)**(mom[0]) *
                                            (beta_lookup[(self.theta[3]+j+1, self.theta[2]+mom[3], 1-initial_freq)]
                                            -beta_lookup[(self.theta[3]+j+1, self.theta[2]+mom[3], initial_freq)])/betaab)
            if mom[0] == 0 and not polarized:
                for j in range(0, mom[3]+1):
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(self.theta[3]+mom[2], self.theta[2]+j+1, 1-initial_freq)]
                                            -beta_lookup[(self.theta[3]+mom[2], self.theta[2]+j+1, initial_freq)])/betaab)
                    out[i] += (binom(mom[3], j) * (-initial_freq)**(mom[3]-j) * (initial_freq)**(mom[1]) *
                                            (beta_lookup[(self.theta[3]+mom[2], self.theta[2]+j+1, initial_freq)])/betaab)
            out[i] *= mc[mom]
        return out/np.sum(out)

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
        rmut = self.rmut_mat.dot(m)

        # Selection
        sel_in = self.sel_mat_in.dot(m_est)
        sel_out = self.sel_mat_out.dot(mp1)

        # Recombination
        rec_out = self.rec_mat_out.dot(m_est)
        rec_in = self.rec_mat_in.dot(mp1)
        rec = rec_in+rec_out

        # Collect all terms
        out = dr + sel_in + sel_out + rec + rmut
        return out


    def integrate_forward_RK45(self, gens, first_gen=0, smallest_step=10**-8, keep_traj = False, time_steps = None, num_points=1, min_step_size=10**-4):
        T = gens/(2*self.pop_size)
        t0 = first_gen/(2*self.pop_size)
        intobj = Integrator(lambda t, x: self.dmdt(t, x, restart_thresh=np.inf), clip=self.clip_ind, renorm=self.renorm)
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
