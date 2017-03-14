import numpy as np
import scipy.linalg
from scipy.linalg import eigh
import h5py
import math
import copy

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn import preprocessing



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
        data_copy=copy.copy(data)
        self.kernel  = make_kernel(data_copy, self.ktype, self.beta)
        return self.kernel

    def Split_Train_Test(self, raw_data):
        kernels = [make_kernel(i, self.ktype, self.beta) for i in raw_data]
        data = [_demean(i) for i in kernels]
        self.test_nodes, self.train_nodes, self.all_nodes = shuffle_nodes(data, self.testPortion, SplitInChunks = 'False')
        if len(self.test_nodes)<=0:
            print "Nothing in the test set!!!"
        print "Training set: "+str(len(self.train_nodes))
        print "Test set: "+str(len(self.test_nodes))
        self.train_by_train = [d[self.train_nodes][:,self.train_nodes] for d in data]           
        self.test_by_train = [d[self.test_nodes][:,self.train_nodes] for d in data]
        self.test_by_test = [d[self.test_nodes][:,self.test_nodes] for d in data]
        self.all_by_train = [d[self.all_nodes][:,self.train_nodes] for d in data]
        #self.all_by_all = [d[self.all_nodes][:,self.all_nodes] for d in data]

        self.raw_train_by_train = [d[self.train_nodes][:,self.train_nodes] for d in raw_data]
        self.raw_test_by_train = [d[self.test_nodes][:,self.train_nodes] for d in raw_data]
        self.raw_test_by_test = [d[self.test_nodes][:,self.test_nodes] for d in raw_data]
        self.raw_all_by_train = [d[self.all_nodes][:,self.train_nodes] for d in raw_data]
        #self.raw_all_by_all = [d[self.all_nodes][:,self.all_nodes] for d in raw_data]

        print "Train by train:", np.shape(self.train_by_train)
        print "Test by train:", np.shape(self.test_by_train)
        print "Test by test:", np.shape(self.test_by_test)
        print "All by train:", np.shape(self.all_by_train)
        #print "All by all:", np.shape(self.all_by_all)
        return self.test_nodes, self.train_nodes, self.train_by_train, self.test_by_train, self.test_by_test, self.all_by_train
        #return self.train_by_train, self.test_by_train, self.test_by_test, self.all_by_train, self.test_nodes, self.train_nodes, self.raw_train_by_train, self.raw_test_by_train, self.raw_test_by_test, self.raw_all_by_train,       
            

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

    def predict_from_features(self, vdata, ws):
        self.proj = np.dot(vdata, ws)
        cov = np.dot(_demean(self.proj), _demean(self.proj).T)
        self.pred=np.zeros((self.proj.shape[0], self.proj.shape[0]))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[0]):
                self.pred[i,j]=cov[i,j]/(math.sqrt(cov[i,i])*math.sqrt(cov[j,j]))
        return self.pred, self.proj

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
        super(CCACrossValidate, self).__init__(numCV = numCV, regs = regs, numCCs = numCCs, kernelcca = kernelcca, ktype = ktype, verbose = verbose, select = select, cutoff = cutoff, testPortion = testPortion, valiPortion = valiPortion, beta=beta)

    def get_hyperPar(self, raw_data, supervise = True):
        """
        Train CCA for a set of regularization coefficients and/or numbers of CCs
        data - list of training data matrices (number of samples X number of features). Number of samples has to match across datasets.
        """
        kernel1=make_kernel(raw_data[0], self.ktype, self.beta)
        kernel2=raw_data[1]
        kernels=[kernel1, kernel2]
        #kernels = [make_kernel(i, self.ktype, self.beta) for i in raw_data]
        data = [_demean(i) for i in kernels]
        print "data:", data[0].shape

        self.corr_mat_train = np.zeros((len(self.regs), len(self.numCCs)))
        self.corr_mat_vali = np.zeros((len(self.regs), len(self.numCCs)))
        self.corr_mat_test = np.zeros((len(self.regs), len(self.numCCs)))
        self.corr_mat_all = np.zeros((len(self.regs), len(self.numCCs)))
        self.auROC_mat_train = np.zeros((len(self.regs), len(self.numCCs)))
        self.auROC_mat_vali = np.zeros((len(self.regs), len(self.numCCs)))
        self.auROC_mat_test = np.zeros((len(self.regs), len(self.numCCs)))
        self.auROC_mat_all = np.zeros((len(self.regs), len(self.numCCs)))
        for ri, reg in enumerate(self.regs):
            for ci, numCC in enumerate(self.numCCs):
                corr_mean_train = 0
                corr_mean_vali = 0
                corr_mean_test = 0
                corr_mean_all = 0
                auROC_mean_train = 0
                auROC_mean_vali = 0
                auROC_mean_test = 0
                auROC_mean_all = 0
                for cvfold in range(self.numCV):
                    if self.verbose:
                        print("Training CV CCA, regularization = %0.4f, %d components, fold #%d" % (reg, numCC, cvfold+1))
                    test_nodes, vali_nodes, train_nodes = shuffle_nodes_intoTVT(data, self.testPortion, self.valiPortion, SplitInChunks = 'False')
                    all_nodes=[i for i in range(data[0].shape[0])]
                    # train
                    ws = kcca([d[train_nodes][:,train_nodes] for d in data], reg, numCC)
                    # predict
                    self.preds_train, self.projs_train = predict([d[train_nodes][:,train_nodes] for d in data], ws)
                    self.preds_vali, self.projs_vali = predict([d[vali_nodes][:,train_nodes] for d in data], ws)
                    self.preds_test, self.projs_test = predict([d[test_nodes][:,train_nodes] for d in data], ws)
                    self.preds_all, self.projs_all = predict([d[all_nodes][:,train_nodes] for d in data], ws)
                    # estimate correlation
                    corr_projs_train = _rowcorr(self.projs_train[0], self.projs_train[1])
                    corr_projs_vali = _rowcorr(self.projs_vali[0], self.projs_vali[1])
                    corr_projs_test = _rowcorr(self.projs_test[0], self.projs_test[1])
                    corr_projs_all = _rowcorr(self.projs_all[0], self.projs_all[1])
                    # estimate auROC
                    fpr_pred_vali, tpr_pred_vali, _ = roc_curve(np.ravel(binarize(raw_data[0])[vali_nodes][:,vali_nodes]), np.ravel(set_diag_to_zero(self.preds_vali[1])))
                    fpr_pred_train, tpr_pred_train, _ = roc_curve(np.ravel(binarize(raw_data[0])[train_nodes][:,train_nodes]), np.ravel(set_diag_to_zero(self.preds_train[1])))
                    fpr_pred_test, tpr_pred_test, _ = roc_curve(np.ravel(binarize(raw_data[0])[test_nodes][:,test_nodes]), np.ravel(set_diag_to_zero(self.preds_test[1])))
                    fpr_pred_all, tpr_pred_all, _ = roc_curve(np.ravel(binarize(raw_data[0])[all_nodes][:,all_nodes]), np.ravel(set_diag_to_zero(self.preds_all[1])))
                    auROC_vali = auc(fpr_pred_vali, tpr_pred_vali)
                    auROC_train = auc(fpr_pred_train, tpr_pred_train)
                    auROC_test = auc(fpr_pred_test, tpr_pred_test)
                    auROC_all = auc(fpr_pred_all, tpr_pred_all)

                    # sum corr for all folds
                    corr_mean_train += np.mean(corr_projs_train)
                    corr_mean_vali += np.mean(corr_projs_vali)
                    corr_mean_test += np.mean(corr_projs_test)
                    corr_mean_all += np.mean(corr_projs_all)
                    # sum auROC for all folds
                    auROC_mean_train += np.mean(auROC_train)
                    auROC_mean_vali += np.mean(auROC_vali)
                    auROC_mean_test += np.mean(auROC_test)
                    auROC_mean_all += np.mean(auROC_all)
                    print "reg=", reg, " numCC=", numCC
                    print "Corr: ", "vali=", np.mean(corr_projs_vali), " train=", np.mean(corr_projs_train), " test=", np.mean(corr_projs_test), " all=", np.mean(corr_projs_all)
                    print "auROC: ", "vali=", auROC_vali, " train=", auROC_train, " test=", auROC_test, " all=", auROC_all
                # devide corr_mat by numCV
                self.corr_mat_vali[ri, ci] = corr_mean_vali/self.numCV
                self.corr_mat_train[ri, ci] = corr_mean_train/self.numCV
                self.corr_mat_test[ri, ci] = corr_mean_test/self.numCV
                self.corr_mat_all[ri, ci] = corr_mean_all/self.numCV
                # devide auROC_mat by numCV
                self.auROC_mat_vali[ri, ci] = auROC_mean_vali/self.numCV
                self.auROC_mat_train[ri, ci] = auROC_mean_train/self.numCV
                self.auROC_mat_test[ri, ci] = auROC_mean_test/self.numCV
                self.auROC_mat_all[ri, ci] = auROC_mean_all/self.numCV
        if supervise is True:
            best_ri, best_ci = np.where(self.auROC_mat_vali == self.auROC_mat_vali.max())
        else:
            best_ri, best_ci = np.where(self.corr_mat_vali == self.corr_mat_vali.max())
        self.best_reg = self.regs[best_ri[0]]
        self.best_numCC = self.numCCs[best_ci[0]]
        # train and predict with best parameters
        ws = kcca([d[train_nodes][:,train_nodes] for d in data], self.best_reg, self.best_numCC)
        self.preds_all, self.projs_all = predict([d[all_nodes][:,train_nodes] for d in data], ws)
        print "Original data:"
        print "Train x Train: ", d[train_nodes][:,train_nodes].shape  
        print "Vali x Vali: ", d[vali_nodes][:,vali_nodes].shape
        print "Test x Test: ", d[test_nodes][:,test_nodes].shape  
        print "All x All: ", d[all_nodes][:,all_nodes].shape 
        print "Projections:"
        print "Train: ", self.projs_train[0].shape 
        print "Vali: ", self.projs_vali[0].shape 
        print "Test: ", self.projs_test[0].shape 
        print "All: ", self.projs_all[0].shape 
        print "Predictions:"
        print "Train x Train: ", self.preds_train[0].shape
        print "Vali x Vali: ", self.preds_vali[0].shape
        print "Test x Test: ", self.preds_test[0].shape
        print "All x All: ", self.preds_all[0].shape 
	return self.best_reg, self.best_numCC, self.preds_all, self.corr_mat_vali, self.corr_mat_train, self.corr_mat_test, self.corr_mat_all, self.auROC_mat_vali, self.auROC_mat_train, self.auROC_mat_test, self.auROC_mat_all

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

def shuffle_nodes_intoTVT(data, testPortion, valiPortion, SplitInChunks = 'False'):
    nT = data[0].shape[0]
    allinds = range(nT)
    if SplitInChunks == 'True':
        chunklen = 10 if nT > 50 else 1
    else:
        chunklen = 1 
    nchunks_test = int(testPortion*nT/chunklen)
    nchunks_vali = int(valiPortion*nT/chunklen)
    indchunks = zip(*[iter(allinds)]*chunklen)
    np.random.shuffle(indchunks)
    test_nodes = [ind for chunk in indchunks[:nchunks_test] for ind in chunk]
    vali_nodes = [ind for chunk in indchunks[nchunks_test:nchunks_vali+nchunks_test] for ind in chunk]
    train_nodes = list(set(allinds)-set(vali_nodes+test_nodes))
    return test_nodes, vali_nodes, train_nodes

def shuffle_data_intoTVT(data, test_nodes, vali_nodes, train_nodes):
    all_nodes=test_nodes+vali_nodes+train_nodes
    train_by_train = [d[train_nodes][:,train_nodes] for d in data]
    test_by_train = [d[test_nodes][:,train_nodes] for d in data]
    vali_by_train = [d[vali_nodes][:,train_nodes] for d in data]
    all_by_train = [d[all_nodes][:,train_nodes] for d in data]
    test_by_test = [d[test_nodes][:,test_nodes] for d in data]
    vali_by_vali = [d[vali_nodes][:,vali_nodes] for d in data]
    return train_by_train, test_by_train, vali_by_train, all_by_train, test_by_test, vali_by_vali    

def set_diag_to_zero(matrix):
    matrix_copy=copy.copy(matrix)
    np.fill_diagonal(matrix_copy, 0)
    return matrix_copy

def binarize(matrix, thres=0):
    matrix_copy=copy.copy(matrix)
    matrix_copy[matrix_copy <= thres] = 0
    matrix_copy[matrix_copy > thres] = 1    
    return matrix_copy
