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
        self.clip_ind = clip_ind
        self.interp_type = interp_type
        self.renorm = renorm
        self.parsimonious = parsimonious
        self.ds_mat = ds_mat
        if self.interp_type == 'loglin' or self.interp_type == 'lin':
            self.vtx, self.wts = self.interp_weights()
        elif self.interp_type == 'jackknife':
            self.gen_jackknife_matrix()
        elif self.interp_type == 'jackknife-constrained':
            self.gen_jackknife_matrix_constrained()
        elif self.interp_type == 'loglinBound':
            self.gen_boundary_interpolator()
        else:
            assert (False), "unknown interpolation type"
        

    @staticmethod
    def makeNonZeroConfigs (slots, daSum):
        daConfigs = []
        size = daSum
        for indices in itertools.combinations(range(1,size), slots - 1):
            indices = (0,) + indices + (size,)
            daConfigs.append (tuple(np.diff(indices)))
        return daConfigs


    @staticmethod
    def findFullRankMatrix (center, initHood, neighbors, tree):
        # try to requery the tree to get more neighbors
        # unfortunately, the best way with the interface seems to be to just do a new search
        dim = len(center)
        currHood = initHood
        currRank = 0
        # first find a large enough set
        while (currRank < dim):
            currNeighbors = tree.query ([center], k=currHood, return_distance=False)[0]
            currRank = np.linalg.matrix_rank (np.transpose (neighbors[currNeighbors]))
            currHood *= 2
        # and then extract a minimal set that gives full rank
        for j in itertools.combinations (currNeighbors, dim):
            lastNeighbors = list(j)
            A = np.transpose (neighbors[lastNeighbors])
            lastRank = np.linalg.matrix_rank (A)
            # only keep if rank improved
            if (lastRank == dim):
                # those neighbors work
                break
        return A  


    @staticmethod
    def fullConfig (K, nonZeroIdxs, compressedConfig):
        realConfig = np.zeros (K, dtype=int)
        for (i, j) in enumerate (list (nonZeroIdxs)):
            realConfig[j] = compressedConfig[i]
        return tuple(realConfig)


    def gen_boundary_interpolator(self):
        # number of genetic types
        K = 4

        # to clip the logit
        self.LOGITCLIP = 300

        # this should get the right levels
        self.daLevels = [x for x in range(K)]

        # compute some normalization factors
        self.normFactor = {}

        for level in self.daLevels:
            self.normFactor[level] = scipy.special.binom (self.order - 1, K - 1 - level) / scipy.special.binom (self.new_order - 1, K - 1 - level)

        # get a dict for the weightmatrices at different levels
        self.weightMatrices = {}

        # and some masks
        self.tallNonZeroMasks = {}

        # I think that we need this as well
        self.smallNonZeroMasks = {}

        daSum = 0
        daHistSum = 0
        for level in self.daLevels:

            # this is the matrix for inter / extra-polation that we have to fill on this level
            thisWeightMatrix = scipy.sparse.lil_matrix ((len(self.moms_new.moms), len(self.moms.moms)))

            # number of elements in config on this level
            numElements = K - level

            smallNonZeroConfigs = self.makeNonZeroConfigs (numElements, self.order)
            tallNonZeroConfigs = self.makeNonZeroConfigs (numElements, self.new_order)

            numEntries = len(tallNonZeroConfigs)
            numZeroPositions = scipy.special.binom (K, level)
            daSum += numZeroPositions * numEntries

            # things add up, so now start getting the actual indices
            # what are the non-zero positions on this level?
            nonZeroPos = [i for i in itertools.combinations (range(K), K-level)]

            # to do the neighborhoods and weights, we need the configurations as frequencies
            # have the same order as configs, so everything good
            smallNonZeroFreqs = np.array (smallNonZeroConfigs, dtype=float)/self.order
            tallNonZeroFreqs = np.array (tallNonZeroConfigs, dtype=float)/self.new_order

            # set up the BallTree with the small ones on this level to find
            tree = sklearn.neighbors.BallTree (smallNonZeroFreqs)
            # and find stuff for the tall ones (number depends on level)
            ind = tree.query (tallNonZeroFreqs, k=numElements, return_distance=False)

            daRanks = []

            # and go through what you find
            for tallIdx, smallIdxs in enumerate(ind):

                # I think that we can do the weights in the recuded vectors
                A = np.transpose (smallNonZeroFreqs[smallIdxs])
                rankA = np.linalg.matrix_rank (A)
                daRanks.append (rankA)
                b = tallNonZeroFreqs[tallIdx]
                assert (A.shape[0] == K - level)
                assert (A.shape[1] == K - level)
                assert (len(b) == K - level)
                
                # rank the matrix
                if (rankA == A.shape[0]):
                    # full rank, all good for linalg.solve
                    pass
                elif (rankA == A.shape[0] - 1):
                    # one rank off, needs work
                    A = self.findFullRankMatrix (tallNonZeroFreqs[tallIdx], 2*numElements, smallNonZeroFreqs, tree)
                    assert (A.shape[0] == K - level)
                    assert (A.shape[1] == K - level)

                    assert (np.linalg.matrix_rank (A) == K - level)
                    # now all good for linalg solve
                else:
                    # should never be more than one rank off
                    assert (False), "Rank of interpolation matrix 2 or more off full rank!"

                # get the weights from linear algebra
                theseWeights = np.linalg.solve (A, b)
                assert (not np.any (np.isnan (theseWeights)))

                # the current indices are not in the full moment vectors
                # so find them in the full moment vectors and put the weights where we want them
                # this one kinda iterates over boundaries
                for (nonIdx, non) in enumerate (nonZeroPos):
                    fullTallConfig = self.fullConfig (K, non, tallNonZeroConfigs[tallIdx])
                    fullTallIdx = self.moms_new.lookup (fullTallConfig)

                    # go through the small indices that we need here
                    for (dI, sI) in enumerate (smallIdxs):
                        fullSmallConfig = self.fullConfig (K, non, smallNonZeroConfigs[sI])
                        fullSmallIdx = self.moms.lookup (fullSmallConfig)

                        # put weights for inter / extra-polation into weight matrix
                        thisWeightMatrix[fullTallIdx, fullSmallIdx] = theseWeights[dI]
                                    
            # get a histogram of the ranks
            maxRank = max(daRanks)
            daHist = np.histogram (daRanks, bins=np.arange(-0.5,maxRank+1.5))
            daHistSum += numZeroPositions * np.sum(daHist[0])

            # and store the weightMatrix for this level
            self.weightMatrices[level] = thisWeightMatrix.tocsr()

            # let's make the nonZeroMasks here, cause it's convenient
            nonZeroIdxs = self.weightMatrices[level].nonzero()
            # first the rows
            nonZeroRows = np.unique(nonZeroIdxs[0])
            nonZeroRowsMask = np.zeros (len(self.moms_new.moms), dtype=bool)
            nonZeroRowsMask[nonZeroRows] = True
            self.tallNonZeroMasks[level] = nonZeroRowsMask
            # then the columns
            nonZeroCols = np.unique(nonZeroIdxs[1])
            nonZeroColsMask = np.zeros (len(self.moms.moms), dtype=bool)
            nonZeroColsMask[nonZeroCols] = True
            self.smallNonZeroMasks[level] = nonZeroColsMask


    ### this one takes a long time
    def isConsistentLoglinBound (self):

        result = True

        # check some consistency
        daLevels = sorted(self.normFactor.keys())

        result &= (daLevels == sorted(self.weightMatrices.keys()))
        result &= (daLevels == sorted(self.tallNonZeroMasks.keys()))
        result &= (daLevels == sorted(self.smallNonZeroMasks.keys()))

        tallDim = self.weightMatrices[daLevels[0]].shape[0]
        smallDim = self.weightMatrices[daLevels[0]].shape[1]

        for level in daLevels:
            result &= (self.weightMatrices[level].shape[0] == tallDim)
            result &= (len(self.tallNonZeroMasks[level]) == tallDim)
            result &= (len(self.smallNonZeroMasks[level]) == smallDim)

            # make sure the weights look good
            aVariable = self.weightMatrices[level].toarray()
            result &= (all (np.isclose (np.sum(aVariable[self.tallNonZeroMasks[level]],axis=1), 1)))
            result &= (all (np.isclose (np.sum(aVariable[~self.tallNonZeroMasks[level]],axis=1), 0)))
            # for the small ones we can do something with columns
            # those should just have something in them
            result &= (all (np.sum (np.abs(aVariable[:,self.smallNonZeroMasks[level]]), axis=0) > 0))
            # those should all be zero
            result &= (all ((np.isclose (aVariable[:,~self.smallNonZeroMasks[level]], 0)).flatten()))

        # if all good we still true
        return result


    ### do the actual interpolation
    def loglinBoundInterp (self, m):
        # so now that we have factors and interpolation matrices for each level, what do?
        # get the small moments
        smallMom = m
        assert (len(m) == len(self.smallNonZeroMasks[0]))

        # and go through the levels to interpolate each individually
        daLevels = sorted(self.normFactor.keys())

        # might actually be good to first do the interpolation on every level seperately, and then see about combining
        interpolates = {}

        smallMasses = np.zeros (len(daLevels))
        tallMasses = np.zeros (len(daLevels))

        for level in daLevels:

            # this could maybe be more efficient, but ok for now
            smallCopy = smallMom.copy()

            # do some weighing
            smallMasses[level] = np.sum (smallCopy[self.smallNonZeroMasks[level]])

            # do weightin on this level
            smallCopy *= self.normFactor[level]

            # logit
            smallLog = scipy.special.logit (smallCopy)

            # I guess we need some clipping here
            # it's not really too nice, but what do zeros mean in the logit interpolation?
            smallLog = np.clip (smallLog, -self.LOGITCLIP, self.LOGITCLIP)

            # and inter/extra-polating?
            tallLog = self.weightMatrices[level].dot (smallLog)

            # so, set the ones that shouldn't have something to not having something
            tallLog[~self.tallNonZeroMasks[level]] = -float("inf")

            # expit
            interpolates[level] = scipy.special.expit (tallLog)

            tallMasses[level] = np.sum (interpolates[level])

        # now how to combine the interpolations on the different levels?
        # I think the way to combine it is to keep some aspect of the shares in smallMasses for tallMasses

        smallLeftSums = np.cumsum (smallMasses[::-1])[::-1]

        massToDistribute = 1

        # and now combine the interpolates with the right factors
        result = np.zeros (len(interpolates[0]))

        for level in daLevels:
            # weigh all that are left to have sum to massToDistribute, weigh according to their fraction of remaining mass on small level
            # on level 0, this should just be 1
            weightFactor = massToDistribute / smallLeftSums[level]

            # now do results on this
            result += weightFactor * interpolates[level]

            # but we ate up some mass, so adjust what we have left to distribute
            massToDistribute = 1 - np.sum (result)

        return result


    # Function to compute the jacknife. I.e. the approximation of M_{n+1} in terms of M_n
    def interp(self, m):
        if self.clip_ind:
            m = np.clip(m, 0, 1)

        if self.interp_type == 'loglin':
            m = np.clip(m, 0, 1)
            logitm = np.clip(scipy.special.logit(m * ((self.order+1) / (self.order+2))**3), -300, 300)
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
        elif self.interp_type == 'jackknife-constrained':
            out= self.jk_mat.dot(m)
        elif self.interp_type == 'loglinBound':
            m = np.clip(m, 0, 1)
            out = self.loglinBoundInterp (m)
        else:
            assert (False), "unknown interpolation type"

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
                    # *gamma(1 + n1)*gamma(1 + n2)*gamma(1 + n3)*gamma(1 + n4))/
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