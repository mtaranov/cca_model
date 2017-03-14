import numpy as np
import scipy.linalg
from scipy.linalg import eigh
import h5py
import math


class _CCABase(object):
    def __init__(self, numCV = None, reg = None, regs = None, numCC = None, numCCs = None, kernelcca = True, ktype = None, verbose = False, select = 0.2, cutoff = 1e-15, beta = 1.0, betas = None, valiPortion = 0.3, testPortion = 0.3):
        self.numCV = numCV
        self.reg = reg
        self.regs = regs
        self.numCC = numCC
        self.numCCs = numCCs
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.select = select
        self.beta = beta
        self.testPortion = testPortion
        self.valiPortion = valiPortion
       # if self.kernelcca and self.ktype == None:
            #self.ktype = "linear"
        self.verbose = verbose


    def kernelize(self, data):
        self.kernel  = make_kernel(data, self.ktype, self.beta)
        return self.kernel

    def Split_Train_Test(self, data):
        self.test_nodes, self.train_nodes, self.all_nodes = shuffle_nodes(data, self.testPortion, SplitInChunks = 'False')
        if len(self.test_nodes)<=0:
            print "Nothing in the test set!!!"
        print "Training set: "+str(len(self.train_nodes))
        print "Test set: "+str(len(self.test_nodes))
        self.train_by_train = [d[self.train_nodes][:,self.train_nodes] for d in data]           
        self.test_by_train = [d[self.test_nodes][:,self.train_nodes] for d in data]
        self.test_by_test = [d[self.test_nodes][:,self.test_nodes] for d in data]
        self.all_by_train = [d[self.all_nodes][:,self.train_nodes] for d in data]
        self.all_by_all = [d[self.all_nodes][:,self.all_nodes] for d in data]
        print "Train by train:", np.shape(self.train_by_train)
        print "Test by train:", np.shape(self.test_by_train)
        print "Test by test:", np.shape(self.test_by_test)
        print "All by train:", np.shape(self.all_by_train)
        print "All by all:", np.shape(self.all_by_all)
        return self.train_by_train, self.test_by_train, self.test_by_test, self.all_by_train, self.test_nodes, self.train_nodes      
            

    def train(self, data):
        print("Training CCA, regularization = %0.4f, %d components" % (self.reg, self.numCC))
        self.ws = kcca(data, self.reg, self.numCC, kernelcca=False)
        return self

    def validate(self, vdata):
        #vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        if not hasattr(self, 'ws'):
            raise NameError("Algorithm needs to be trained!")
        self.preds, self.projs = predict(vdata, self.ws)
        return self.preds, self.projs

    def compute_ev(self, data):
        nD = len(data)
        nT = data[0].shape[0]
        self.numCC = nT if self.numCC is None else self.numCC
        nF = [d.shape[1] for d in data]
        self.ev = [np.zeros((self.numCC, f)) for f in nF]
        for cc in range(self.numCC):
            ccs = cc+1
            if self.verbose:
                print("Computing explained variance for component #%d" % ccs)
            preds, corrs = predict(data, [w[:, ccs-1:ccs] for w in self.ws], self.cutoff)
            resids = [abs(d[0]-d[1]) for d in zip(data, preds)]
            for s in range(nD):
                ev = abs(data[s].var(0) - resids[s].var(0))/data[s].var(0)
                ev[np.isnan(ev)] = 0.
                self.ev[s][cc] = ev
        return self.ev

    def save(self, filename):
        h5 = h5py.File(filename, "a")
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, list):
                    for di in range(len(value)):
                        grpname = "dataset%d" % di
                        dgrp = h5.require_group(grpname)
                        try:
                            dgrp.create_dataset(key, data=value[di])
                        except RuntimeError:
                            del h5[grpname][key]
                            dgrp.create_dataset(key, data=value[di])
                else:
                    h5.attrs[key] = value
        h5.close()

    def load(self, filename):
        h5 = h5py.File(filename, "a")
        for key, value in h5.attrs.items():
            setattr(self, key, value)
        for di in range(len(h5.keys())):
            ds = "dataset%d" % di
            for key, value in h5[ds].items():
                if di == 0:
                    setattr(self, key, [])
                self.__getattribute__(key).append(value.value)

class CCACrossValidate(_CCABase):
    '''Attributes:
        numCV - number of crossvalidation folds
        reg - array of regularization parameters. Default is np.logspace(-3, 1, 10)
        numCC - list of numbers of canonical dimensions to keep. Default is np.range(5, 10).
        kernelcca - True if using a kernel (default), False if not kernelized.
        ktype - type of kernel if kernelcca == True (linear or gaussian). Default is linear.
        verbose - True is default

    Results:
        ws - canonical weights
        comps - canonical components
        cancorrs - correlations of the canonical components on the training dataset
        corrs - correlations on the validation dataset
        preds - predictions on the validation dataset
        ev - explained variance for each canonical dimension
    '''
    def __init__(self, numCV = None, regs = None, numCCs = None, kernelcca = True, ktype = None, verbose = True, select = 0.3, cutoff = 1e-15, betas = None, testPortion = 0.3, valiPortion = 0.3, beta=1):
        numCV = 10 if numCV is None else numCV
        regs = np.array(np.logspace(-3, 1, 10)) if regs is None else np.array(regs)
        numCCs = np.arange(5, 10) if numCCs is None else np.array(numCCs)
        betas = np.array([1]) if beta is None else np.array(betas)
        super(CCACrossValidate, self).__init__(numCV = numCV, regs = regs, numCCs = numCCs, kernelcca = kernelcca, ktype = ktype, verbose = verbose, select = select, cutoff = cutoff, testPortion = 0.3, valiPortion = 0.3, beta=1)



    def get_hyperPar(self, data):
        """
        Train CCA for a set of regularization coefficients and/or numbers of CCs
        data - list of training data matrices (number of samples X number of features). Number of samples has to match across datasets.
        """
        self.corr_mat = np.zeros((len(self.regs), len(self.numCCs)))
        for ri, reg in enumerate(self.regs):
            for ci, numCC in enumerate(self.numCCs):
                corr_mean = 0
                for cvfold in range(self.numCV):
                    if self.verbose:
                        print("Training CV CCA, regularization = %0.4f, %d components, fold #%d" % (reg, numCC, cvfold+1))
                    heldinds, notheldinds, allinds = shuffle_nodes(data, self.valiPortion, SplitInChunks = 'False')
                    ws = kcca([d[notheldinds][:,notheldinds] for d in data], reg, numCC)
                    self.preds, self.projs = predict([d[heldinds][:,notheldinds] for d in data], ws)
                    self.preds_all, self.projs_all = predict([d[allinds][:,notheldinds] for d in data], ws)
                    corr_projs = _rowcorr(self.projs[0], self.projs[1])
                    corr_projs_all = _rowcorr(self.projs_all[0], self.projs_all[1])
                    corr_mean += np.mean(corr_projs)
                    #for d in data:
                       # print d[notheldinds][:,notheldinds].shape  
                        #print d[heldinds][:,heldinds].shape
                        #print d[heldinds][:,notheldinds].shape  
                        #print self.preds[0].shape, self.preds[1].shape,  self.projs[0].shape, self.projs[1].shape
                    print "reg=", reg, " numCC=", numCC, " corr_projs_vali=", np.mean(corr_projs), " corr_projs_train=", np.mean(corr_projs_all)
                    self.train_nodes = notheldinds
                    self.vali_nodes = heldinds
                    self.all_nodes=allinds    
                self.corr_mat[ri, ci] = corr_mean/self.numCV
        best_ri, best_ci = np.where(self.corr_mat == self.corr_mat.max())
        self.best_reg = self.regs[best_ri[0]]
        self.best_numCC = self.numCCs[best_ci[0]]
        #self.comps = kcca(data, self.best_reg, self.best_numCC, kernelcca = self.kernelcca, ktype = self.ktype)
        #self.ws, self.cancorrs = recon(data, self.comps, kernelcca = self.kernelcca)
        #return self.preds, self.projs, self.train_nodes, self.test_nodes, self.all_nodes, self.preds_all, self.projs_all, self.corr_mat, self.best_reg, self.best_numCC
	#return self.corr_mat, self.best_reg, self.best_numCC
	return self.corr_mat, self.best_reg, self.best_numCC, self.preds_all, self.projs_all, self.preds, self.projs

class CCA(_CCABase):
    '''Attributes:
        reg - regularization parameters. Default is 0.1.
        numCC - number of canonical dimensions to keep. Default is 10.
        kernelcca - True if using a kernel (default), False if not kernelized.
        ktype - type of kernel if kernelcca == True (linear or gaussian). Default is linear.
        verbose - True is default

    Results:
        ws - canonical weights
        comps - canonical components
        cancorrs - correlations of the canonical components on the training dataset
        corrs - correlations on the validation dataset
        preds - predictions on the validation dataset
        ev - explained variance for each canonical dimension
    '''
    def __init__(self, reg = 0.1, numCC = 10, kernelcca = True, ktype = None, verbose = True, cutoff = 1e-15, beta = 1.0, testPortion = 0.3):
        super(CCA, self).__init__(reg = reg, numCC = numCC, kernelcca = kernelcca, ktype = ktype, verbose = verbose, cutoff = cutoff, beta = 1.0, testPortion = 0.3)

    def train(self, data):
        return super(CCA, self).train(data)

def predict(vdata, ws):
    projs = _listdot([d.T for d in vdata], ws)
    preds = []

    for proj in projs:
        cov = np.dot(_demean(proj), _demean(proj).T)
        pred=np.zeros((proj.shape[0], proj.shape[0]))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[0]):
                pred[i,j]=cov[i,j]/(math.sqrt(cov[i,i])*math.sqrt(cov[j,j]))
        preds.append(pred)
        #cs = np.nan_to_num(_rowcorr(vdata[dnum].T, pred.T))
        #corrs.append(cs)
    #return preds, corrs, projs
    return preds, projs

def kcca(data, reg = 0.1, numCC=None, kernelcca = False, returncorrs = False):
    '''Set up and solve the eigenproblem for the data in kernel and specified reg
    '''
    kernel = [d.T for d in data]

    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

    # Get the kernel auto- and cross-covariance matrices
    crosscovs = [np.dot(ki, kj.T).T for ki in kernel for kj in kernel]

    # Allocate LH and RH:
    LH = np.zeros((np.sum(nFs), np.sum(nFs)))
    RH = np.zeros((np.sum(nFs), np.sum(nFs)))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(len(kernel)):
        RH[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1])), int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1]))] = crosscovs[i*(len(kernel)+1)] + reg*np.eye(nFs[i])
        for j in range(len(kernel)):
            if i !=j:
                LH[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1])), int(np.sum(nFs[:j])):int(np.sum(nFs[:j+1]))] = crosscovs[len(kernel)*j+i]

    LH = (LH+LH.T)/2.
    RH = (RH+RH.T)/2.

    r, Vs = eigh(LH, RH)

    if kernelcca:
        comp = []
        for i in range(len(kernel)):
            comp.append(Vs[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1]))])
        tcorrs = recon(data, comp, corronly = True, kernelcca = kernelcca)
        tc = [t[0, 1] for t in tcorrs]
        i = np.argsort(tc)[::-1]
        comp = [c[:, i[:numCC]] for c in comp]
    else:
        # Get vectors for each dataset
        r[np.isnan(r)] = 0
        rindex = np.argsort(r)[::-1]
        r = r[rindex]
        comp = []
        Vs = Vs[:, rindex]
        rs = np.sqrt(r[:numCC]) # these are not correlations for kernel CCA with regularization

        for i in range(len(kernel)):
            comp.append(Vs[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1])), :numCC])
    if returncorrs:
        if kernelcca:
            return comp, tcorrs
        else:
            return comp, rs
    else:
        return comp

def recon(data, comp, corronly=False, kernelcca = True):
    nT = data[0].shape[0]
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return ws, corrs

#def _zscore(d): return (d-d.mean(0))/d.std(0)
def _zscore(d): return (d-d.mean(0))/d.std(0)
def _demean(d): return d-d.mean(0)
def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]
def _listcorr(a):
    '''Returns pairwise row correlations for all items in array as a list of matrices
    '''
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j>i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0,1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs
def _rowcorr(a, b):
    '''Correlations between corresponding matrix rows
    '''
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0,1]
    return cs

def make_kernel(d, ktype = "diffusion", beta = 1.0):
#def make_kernel(d, normalize = True, ktype = "diffusion", sigma = 1.0, beta = 1.0):
    '''Makes a kernel for data d
      If ktype is "diffusion", the kernel is a diffusion kernel
      If ktype is "linear", the kernel is a linear inner product
      If ktype is "gaussian", the kernel is a Gaussian kernel with sigma = sigma
    '''
    if ktype == "diffusion":
        # 1.Computes Degree matrix  - diagonal matrix with diagonal entries = raw sums of adjacency matrix 
        DegreeMatrix = np.diag(np.sum(d, axis=0))
        # 2. Computes negative Laplacian H = AdjMatrix - DegreeMatrix
        H = np.subtract(d, DegreeMatrix)
        # 3. Computes matrix exponential: exp(beta*H)
        kernel = scipy.linalg.expm(beta*H)

#    if ktype == "linear":
#        d = np.nan_to_num(d)
#        cd = _demean(d)
#        kernel = np.dot(cd,cd.T)
#        kernel = (kernel+kernel.T)/2.
#        kernel = kernel / np.linalg.eigvalsh(kernel).max()
#    elif ktype == "gaussian":
#        from scipy.spatial.distance import pdist, squareform
#        # this is an NxD matrix, where N is number of items and D its dimensionalites
#        pairwise_dists = squareform(pdist(d, 'euclidean'))
#        kernel = np.exp(-pairwise_dists ** 2 / sigma ** 2)
#        kernel = (kernel+kernel.T)/2.
#        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel


def shuffle_nodes(data, testPortion, SplitInChunks = 'False'):
    nT = data[0].shape[0]
    allinds = range(nT)
    if SplitInChunks == 'True':
        chunklen = 10 if nT > 50 else 1
    else:
        chunklen = 1
    nchunks = int(testPortion*nT/chunklen)
    indchunks = zip(*[iter(allinds)]*chunklen)
    np.random.shuffle(indchunks)
    heldinds = [ind for chunk in indchunks[:nchunks] for ind in chunk]
    notheldinds = list(set(allinds)-set(heldinds))
    return heldinds, notheldinds, allinds
