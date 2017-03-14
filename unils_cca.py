import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.mstats import mquantiles
import scipy.stats as ss
import math
import scipy.linalg
import itertools
import copy
import random
import gzip
import rcca_modified  as rcca
reload(rcca)

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform

# builds adjacency matrix 
def BuildMatrixA(PromoterFile, InteractionsFile, datatype, chr):

    REFrag_dict={}
    index=0
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        key=(words[0], words[1], words[2])
        if words[0] in [chr]: # only chr1
           REFrag_dict[key]=index
           index+=1

    # Initialize matrix (promoter x promoter)
    PPMatrix=np.zeros((len(REFrag_dict), len(REFrag_dict))) #  number of promoters in chr 1

    # Fill (promoter x promoter) matrix with q-values of promoter-promoter interaction
    max_score=0
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        if float(words[6])!=0:  
            if (-1)*math.log10(float(words[6]))>max_score:
                max_score=(-1)*math.log10(float(words[6])) 
            
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        if words[0] in [chr]:
            i=REFrag_dict[(words[0], words[1], words[2])]
            j=REFrag_dict[(words[3], words[4], words[5])]
            if datatype == 'HiC':
                if float(words[6])!=0:                        
                    q_values=(-1)*math.log10(float(words[6])) # for HiC
                else:
                     q_values=max_score
            else:
                q_values=float(words[6])  # for CaptureC
            if PPMatrix[i,j] != 0:
                PPMatrix[i,j]=PPMatrix[i,j]/2+q_values/2
                PPMatrix[j,i]=PPMatrix[j,i]/2+q_values/2
            else:
                PPMatrix[i,j]=q_values
                PPMatrix[j,i]=q_values
                    # take -1*log(Q) for non-zero entries
    #mask = PPMatrix != 0
    #PPMatrix[mask] = np.log10(PPMatrix[mask])*(-1)
    return PPMatrix

def print_q_val_hist(PPMatrix):
    # list of non-zero q-values
    q_values=list(filter((0.0).__ne__,list(itertools.chain.from_iterable(np.array(PPMatrix).tolist()))))

    # Some tests:
    print "Some tests on adjacency matrix:"
    # 1. Check if the matrix is symmetric:
    if (PPMatrix.transpose() == PPMatrix).all() == True:
        print "Adjacency matrix is symmetric"
    # 2. Print out average q-values:
    print "Average q-value with zeros: ", str(np.average(PPMatrix))
    print "Average q-value w/o zeros: ", np.mean(q_values)

    # Print distribution of q-values
    plt.hist(q_values)
    plt.show()

def DiffusionKernel(AdjMatrix, beta):
    # 1.Computes Degree matrix  - diagonal matrix with diagonal entries = raw sums of adjacency matrix 
    DegreeMatrix = np.diag(np.sum(AdjMatrix, axis=0))
    # 2. Computes negative Laplacian H = AdjMatrix - DegreeMatrix
    H = np.subtract(AdjMatrix, DegreeMatrix)
    # 3. Computes matrix exponential: exp(beta*H)
    K = scipy.linalg.expm(beta*H)

   # tests:
   # plot cummulative (1-q)-value (raw sums) for all promoters
    plt.plot(np.sum(AdjMatrix, axis=1))
    plt.xlim(0, len(AdjMatrix))
    plt.show()
    #return K / np.linalg.eigvalsh(K).max()
    return K

def plot_zscore(matrix, label, title):
    matrix_copy=copy.copy(matrix)
    printMatrix(zscore(matrix_copy), label, 1, 1, title)
    

def printMatrix(Matrix, ylabel, QuantileValue, LowerUpperLimit, chr):
    title = chr
    #vmaxLim=mquantiles(Matrix,[0.99])[0]
    Lim=mquantiles(Matrix,[QuantileValue])[0]
    print Matrix.max()
    print np.shape(Matrix)
    print "Limit:", Lim
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    if LowerUpperLimit == 'lower':
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmin=Lim)
    else:
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmax=Lim) # cmap="RdYlBu_r"


    ax.axhline(-0.5, color="#000000", linewidth=1, linestyle="--")
    ax.axvline(-0.5, color="#000000", linewidth=1, linestyle="--")

    cb = fig.colorbar(m)
    cb.set_label(ylabel)

    ax.set_ylim((-0.5, len(Matrix) - 0.5))
    ax.set_xlim((-0.5, len(Matrix) - 0.5))
    
    plt.title(title)
    plt.show()
    return

# remove mean in feature space
def center(X, scale=False):
    """ Center X and scale if the scale parameter==True
    Returns X
    """
    if scale:
        return (X-X.mean(0))/X.std(0)
    else:
        return X-X.mean(0)

def diagNorm(square_matrix):
    return square_matrix / np.linalg.eigvalsh(square_matrix).max()

#test proportion is computed as total-train
def shuffle_nodes(PPMatrix, trainProportion):
    total_num=len(PPMatrix)
    train_num=int(trainProportion*total_num)
    test_num=total_num-train_num
    if test_num<=0:
        print "Nothing in the test set!!!"
    print "Training set: "+str(train_num)
    print "Test set: "+str(test_num)
    #decide the random split of nodes
    nodes = [i for i in range(len(PPMatrix))]
    shuffled_nodes=copy.copy(nodes)
    random.shuffle(shuffled_nodes)
    train_nodes=np.array(shuffled_nodes[:train_num])
    test_nodes=np.array(shuffled_nodes[train_num:])
    shuffled_nodes=[train_nodes, test_nodes]
    return shuffled_nodes
#split each dataset based on the decided split
def train_test_2D(PPMatrix, trainProportion):
    shuffled_nodes = shuffle_nodes(PPMatrix, trainProportion)
    train_by_train=PPMatrix[shuffled_nodes[0]][:,shuffled_nodes[0]]
    test_by_train=PPMatrix[shuffled_nodes[1]][:,shuffled_nodes[0]]
    test_by_test=PPMatrix[shuffled_nodes[1]][:,shuffled_nodes[1]]
    all_by_train=PPMatrix[np.hstack((shuffled_nodes[0], shuffled_nodes[1]))][:,shuffled_nodes[0]]
    all_by_all=PPMatrix[np.hstack((shuffled_nodes[0], shuffled_nodes[1]))][:,np.hstack((shuffled_nodes[0], shuffled_nodes[1]))]
    #print "Train by train:", np.shape(train_by_train), "Test by train:", np.shape(test_by_train), "Test by test:", np.shape(test_by_test), "All by train:", np.shape(all_by_train), "All by all:", np.shape(all_by_all) 
    print "Train by train:", np.shape(train_by_train),  "Test by test:", np.shape(test_by_test), "All by all:", np.shape(all_by_all) 
#       #def scaleData(data):
#         #   return preprocessing.scale(data)
#    #return scaleData(train), scaleData(vali), scaleData(test)
#    return train_by_train, test_by_train, test_by_test, all_by_train, all_by_all
    return train_by_train, test_by_test, shuffled_nodes[0], shuffled_nodes[1]

def run_cca(data, test, numCC,reg, title):
    cca = rcca.CCA(kernelcca=False, numCC=numCC, reg=reg)
    cca.train(data)
    # Find canonical components
    # Test on held-out data
    corrs = cca.validate(test)
    print "Pearson Correlation: ", sum(corrs[0])/len(corrs[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot0 = ax.bar(np.arange(corrs[0].shape[0]), corrs[0], 0.3, color = "steelblue")
    #plot1 = ax.bar(np.arange(corrs[1].shape[0])+0.35, corrs[1], 0.3, color="orangered")
    #ax.legend([plot0[0], plot1[0]], ["Dataset 1", "Dataset 2"])
    ax.set_title(title) 
    ax.set_ylabel("Prediction correlation")
    #ax.set_xticks(np.arange(0, corrs[0].shape[0], 20)+0.325)
    #ax.set_xticklabels(["%d" % i for i in range(0, corrs[0].shape[0], 20)])
    ax.set_xlabel("Test data")
#    fig.savefig(str(PROJDIR)+'/Prediction.png', dpi=fig.dpi)
    return cca.train(data)

def run_cca_ppi(data, test, numCC,reg, title):
    cca = rcca.CCA(kernelcca=False, numCC=numCC, reg=reg)
    cca.train(data)
    # Find canonical components
    # Test on held-out data
    preds, projs = cca.validate_ppi(test)

    #return cca.train(data)
    return preds, projs

def ScatterPlot(cca_train):
    colors=['red','blue','green']
    len_comps=len(cca_train.__dict__['comps'])
    for comp in range(len_comps):
        print cca_train.__dict__['comps'][comp].shape #the components from the "comp"st dataset
        #Plot them on the same plot
        for i in range(len(cca_train.__dict__['comps'][comp][0])-1):
            plt.scatter(cca_train.__dict__['comps'][comp][:,i], cca_train.__dict__['comps'][comp][:,i+1],c=colors[comp])
            plt.title('CC '+str(i+1)+' vs'+' CC '+str(i+2))
            plt.show()

# builds feature vector 
def BuildFeatureVector(PromoterFile, FeatureVectorFile, chr):

    REFrag_dict={}
    index=0
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        key=(words[0], words[1], words[2])
        if words[0] in [chr]: # only chr1
           REFrag_dict[key]=index
           index+=1

    # Initialize vector (promoters only)
    vector=np.zeros((len(REFrag_dict),1)) #  number of promoters in chr 1

    for line in open(FeatureVectorFile,'r'):
        words=line.rstrip().split()
        if words[0] in [chr]: #only chr1
            value=words[3]
            i=REFrag_dict[(words[0], words[1], words[2])]
            vector[i]=value
    return vector

def print_feature_hist(vector, dataName):
    # list of non-zero q-values
    nonzero_values=filter(lambda a: a != 0, np.array(vector).reshape(-1,).tolist())

    # Print out average q-values:
    print "Average value with zeros: ", str(np.average(vector))
    print "Average q-value w/o zeros: ", np.mean(nonzero_values)

    # Print distribution of q-values
    plt.hist(vector)
    plt.title(str(dataName))
    plt.show()


def PearsonCorr(a,b):
    '''Correlations between corresponding matrix rows
    '''
    a=a.T
    b=b.T
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0,1]
    return np.nan_to_num(cs)

def make_kernel(d, ktype = "linear", sigma = 1.0):
    '''Makes a kernel for data d
      If ktype is "linear", the kernel is a linear inner product
      If ktype is "gaussian", the kernel is a Gaussian kernel with sigma = sigma
    '''
    if ktype == "linear":
        d = np.nan_to_num(d)
        cd = demean(d)
        kernel = np.dot(cd,cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform
        # this is an NxD matrix, where N is number of items and D its dimensionalites
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / sigma ** 2)
    kernel = (kernel+kernel.T)/2.
    kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel

# builds adjacency matrix 
def reconstructCaptureC(indx1, indx2, matrix, matrixSize):

    # Initialize matrix (promoter x promoter)
    PPMatrix=np.zeros((matrixSize, matrixSize)) #  number of promoters in chr 1

    # Fill (promoter x promoter) matrix with q-values of promoter-promoter interaction
    m=0
    for i in indx1:
        n=0
        for j in indx2:
            PPMatrix[i,j]=matrix[m,n]
            n+=1
        m+=1    
    #for symmetric matricies: train x train, test x test
    if len(indx1)==len(indx2):
        PPMatrix=(PPMatrix+PPMatrix.T)/2
    #for non-symmetric matricies: test x train
    else:
        PPMatrix=PPMatrix+PPMatrix.T

          
    # list of non-zero q-values
    q_values=list(filter((0.0).__ne__,list(itertools.chain.from_iterable(np.array(PPMatrix).tolist()))))

    # Some tests:
    print "Some tests on adjacency matrix:"
    # 1. Check if the matrix is symmetric:
    if (PPMatrix.transpose() == PPMatrix).all() == True:
        print "Adjacency matrix is symmetric"
    # 2. Print out average q-values:
    print "Average q-value with zeros: ", str(np.average(PPMatrix))
    print "Average q-value w/o zeros: ", np.mean(q_values)

    # Print distribution of q-values
#    plt.hist(q_values)
#    plt.show()

    return PPMatrix

def set_diag_to_zero(matrix):
    np.fill_diagonal(matrix, 0)
    return matrix

def binarize(matrix, thres=0):
    matrix_copy = copy.copy(matrix)
    matrix_copy[matrix <= thres] = 0
    matrix_copy[matrix > thres] = 1    
    return matrix_copy

def binarize_wAmbig(matrix, thres=0):
    matrix_copy = copy.copy(matrix)
    matrix_copy[matrix == 0] = 0
    matrix_copy[matrix > thres] = 1    
    return matrix_copy

def pred_from_ppi(cca_train):
    proj=cca_train.__dict__['ccomp'][0]
    cov=np.dot(center(proj), center(proj).T)
    corr=np.zeros((cca_train.__dict__['ccomp'][0].shape[0], cca_train.__dict__['ccomp'][0].shape[0]))
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            corr[i,j]=cov[i,j]/(math.sqrt(cov[i,i])*math.sqrt(cov[j,j]))
    #if (corr.transpose() == corr).all() == True:
    #    print "matrix is symmetric"
    return corr

def demean(d): return d-d.mean(0) 
def zscore(d): return (d-d.mean(0))/d.std(0)

def build_distance_matrix(PromoterFile, chr):
    REsiteMids=[]
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        if words[0] in [chr]: # only chr1
            REsiteMids.append((int(words[2])+int(words[1]))/2)
    distance_matrix=np.zeros((len(REsiteMids), len(REsiteMids)))
    for i in range(len(REsiteMids)):
        for j in range(len(REsiteMids)):
                distance_matrix[i,j]=abs(REsiteMids[i]-REsiteMids[j])
    np.fill_diagonal(distance_matrix, 0.1)
    return distance_matrix

def build_distance_for_node(PromoterFile): 
    REFrag_dict={}
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        if chr not in REFrag_dict:
            index=0
            REFrag_dict[chr]={hind3:index}
        else:
            index+=1
        REFrag_dict[chr].update({hind3:index})

    REsiteMids={}
    for chr in REFrag_dict:
        REsiteMids[chr] = np.zeros((len(REFrag_dict[chr]),1))
        for key in REFrag_dict[chr]:
            start = int(key[0])
            end = int(key[1])
            REsiteMids[chr][REFrag_dict[chr][key]] = (start + end)/2
    return REsiteMids

# builds feature vector 
def get_features(PromoterFile, FeatureVectorFile, dataName):

    REFrag_dict={}
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        if chr not in REFrag_dict:
            index=0
            REFrag_dict[chr]={hind3:index}
        else:
            index+=1
        REFrag_dict[chr].update({hind3:index})

    # Initialize vector (promoters only)
    vector=np.zeros((len(REFrag_dict),)) #  number of promoters in chr 1

    features={}
    for line in open(FeatureVectorFile,'r'):              
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        value = words[3]
        if chr not in features:
            # Initialize vector (promoters only)
            features[chr] = np.zeros((len(REFrag_dict[chr]))) #  number of promoters in chr 
            features[chr][REFrag_dict[chr][hind3]] = value
        else:
            features[chr][REFrag_dict[chr][hind3]] = value
            

    # list of non-zero q-values
    nonzero_values=filter(lambda a: a != 0, np.array(features['chr1']).reshape(-1,).tolist()) 

    # Print out average q-values:
    print "Average value with zeros: ", str(np.average(features['chr1']))
    print "Average q-value w/o zeros: ", np.mean(nonzero_values)

    # Print distribution of q-values
    plt.hist(features['chr1'])
    plt.title(str(dataName))
    plt.show()

    return features

def reshape_2d_to_1d(matrix):
    return matrix.ravel()

def remove_ambiguous(labels, preds, pair_dist,  thres = 10):
    if labels.shape != preds.shape or labels.shape != pair_dist.shape:
        raise ValueError('labels, preds and pair_dist are not the same dimensions!')

    neg_indxs = np.where(labels.astype(float)==0)[0]
    pos_indxs = np.where(labels.astype(float)>thres)[0]

    labels_pos=labels[pos_indxs]
    labels_neg=labels[neg_indxs]
    preds_pos=preds[pos_indxs]
    preds_neg=preds[neg_indxs]
    pair_dist_pos=pair_dist[pos_indxs]
    pair_dist_neg=pair_dist[neg_indxs]

    labels_new=np.concatenate((labels_pos, labels_neg))
    preds_new=np.concatenate((preds_pos, preds_neg))
    pair_dist_new=np.concatenate((pair_dist_pos, pair_dist_neg))
    
    return labels_new, preds_new, pair_dist_new
    #return labels_pos, labels_neg



def get_pairs_distance_matched(labels_all, preds_all,  pair_dist_all,  min_dist, max_dist, dist_step, imbalance_ratio, thres):
    # remove ambigous labels - larger than 0, lower than thres
    labels, preds,  pair_dist = remove_ambiguous(labels_all, preds_all,  pair_dist_all,  thres=thres)

    def subsample_indx(indecies, size, imbalance_ratio):
        indecies_shuffled=copy.copy(indecies)
        np.random.shuffle(indecies_shuffled)
        num_subsampled = size*imbalance_ratio
        if num_subsampled > len(indecies[0]):
            print '    Error: Not enough to subsample'
            exit
        #print indecies_shuffled[0].shape
        #print indecies_shuffled[0][:num_subsampled].shape
        else:
            return indecies_shuffled[0][:num_subsampled]

    neg_indxs = np.where(labels.astype(float)==0)[0]
    pos_indxs = np.where(labels.astype(float)>thres)[0]
    labels_pos=labels[pos_indxs]
    labels_neg=labels[neg_indxs]
    preds_pos=preds[pos_indxs]
    preds_neg=preds[neg_indxs]
    pair_dist_pos=pair_dist[pos_indxs]
    pair_dist_neg=pair_dist[neg_indxs]

    thres1=min_dist+dist_step
    thres2=min_dist

    labels_new=np.empty(([0,]))
    preds_new=np.empty(([0,]))
    pair_dist_new=np.empty(([0,]))

    while thres1 <= max_dist:
        print 'distance window: ', '[', thres2, ',', thres1, ']'
        neg_indx_at_dist=np.where((pair_dist_neg[:,].astype(int) <= thres1) & (pair_dist_neg[:,].astype(int) >= thres2))
        pos_indx_at_dist=np.where((pair_dist_pos[:,].astype(int) <= thres1) & (pair_dist_pos[:,].astype(int) >= thres2))
        if len(pos_indx_at_dist[0])> len(neg_indx_at_dist[0]):
            print 'more pos than neg'
            print 'number of pos at distance=',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(pos_indx_at_dist, len(neg_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=indx_subsampled
            new_neg_indx_at_dist=neg_indx_at_dist[0]

        else:
            #print 'more neg than pos'
            print 'number of pos at distance=',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(neg_indx_at_dist, len(pos_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=pos_indx_at_dist[0]
            new_neg_indx_at_dist=indx_subsampled

        labels_pos_at_dist=labels_pos[new_pos_indx_at_dist]
        labels_neg_at_dist=labels_neg[new_neg_indx_at_dist]
        preds_pos_at_dist=preds_pos[new_pos_indx_at_dist]
        preds_neg_at_dist=preds_neg[new_neg_indx_at_dist]
        #pair_dist_pos_at_dist=pair_dist_pos[new_pos_indx_at_dist]
        #pair_dist_neg_at_dist=pair_dist_neg[new_neg_indx_at_dist]

        labels_at_dist=np.concatenate((labels_pos_at_dist, labels_neg_at_dist))
        preds_at_dist=np.concatenate((preds_pos_at_dist, preds_neg_at_dist))
        #pair_dist_at_dist=np.concatenate((pair_dist_pos_at_dist, pair_dist_neg_at_dist))

        print 'labels at dist: ', labels_at_dist.shape
        print 'preds at dist: ', preds_at_dist.shape
        #print 'pair_dist at dist: ', pair_dist_at_dist.shape

        labels_new=np.concatenate((labels_new, labels_at_dist))
        preds_new=np.concatenate((preds_new, preds_at_dist))
        #pair_dist_new=np.concatenate((pair_dist_new, pair_dist_at_dist))


        #print X_new.shape, X_at_dist.shape
        #print y_new.shape, y_at_dist.shape
        #print indx_new.shape, indx_at_dist.shape

        #print "# of neg:", np.where(y_at_dist==0)[0].shape
        #print "# of pos:", np.where(y_at_dist==1)[0].shape
        #thres2=thres1+min_dist
        thres2=thres1
        thres1=thres1+dist_step

    return np.expand_dims(binarize(labels_new, thres=thres).astype(int), axis=1), np.expand_dims(preds_new, axis=1)

def BuildMatrix_w_ambig(PromoterFile, InteractionsFile):

    REFrag_dict={}
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        if chr not in REFrag_dict:
            index=0
            REFrag_dict[chr]={hind3:index}
        else:
            index+=1
        REFrag_dict[chr].update({hind3:index})
        
    labels_score={}
    for chr in REFrag_dict:
        uniq=0.0
        non_uniq=0.0
        # Initialize matrix (promoter x promoter)
        labels_score[chr] = np.ones((len(REFrag_dict[chr]), len(REFrag_dict[chr]))) #  number of promoters in chr 
        labels_score[chr] = labels_score[chr]*(-1)
 
    total_lines=0.0
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3_1 = (words[1], words[2])
        hind3_2 = (words[4], words[5])
        q_values = float(words[6])

        i=REFrag_dict[chr][hind3_1]
        j=REFrag_dict[chr][hind3_2]
        
        if labels_score[chr][i,j] != -1:
            non_uniq+=1
            # mean
            labels_score[chr][i,j]=labels_score[chr][i,j]/2+q_values/2
            labels_score[chr][j,i]=labels_score[chr][j,i]/2+q_values/2
#             # max
#             labels_score[chr][i,j]=max(labels_score[chr][i,j], q_values)
#             labels_score[chr][j,i]=max(labels_score[chr][j,i], q_values)
        else:
            uniq +=1
            labels_score[chr][i,j]=q_values
            labels_score[chr][j,i]=q_values
            
        total_lines += 1
    
    print "non-unique entries in CaptureC file(bait1-bait2/bait2-bait1): ", non_uniq, " ", non_uniq/total_lines, "%"
    print "unique entries in CaptureC file(bait1-bait2/bait2-bait1): ", uniq, " ", uniq/total_lines, "%"
    return labels_score

def get_pos_neg_1d(data_1d, thres=0):
    neg_indxs = np.where(data_1d.astype(float)==0)[0]
    pos_indxs = np.where(data_1d.astype(float)>thres)[0]
    indx = np.concatenate((neg_indxs, pos_indxs))
    new_data_1d = data_1d[indx]
    return new_data_1d, indx

def get_pos_neg_y_and_ypred_at_thres(labels, preds, thres=0):
    data_1d = labels.reshape(labels.shape[0]*labels.shape[0],1)
    labels_pos_neg, indx = get_pos_neg_1d(data_1d, thres=thres)
    y = binarize(labels_pos_neg).astype(int)
    pred_1d = preds.reshape(preds.shape[0]*preds.shape[0],1)
    y_pred = pred_1d[indx]
    return y, y_pred, indx  

def subsample_neg(y, y_pred, imbalance_ratio, thres=0):
    neg_indxs = np.where(y.astype(float)==0)[0]
    pos_indxs = np.where(y.astype(float)>thres)[0]
    size = pos_indxs.shape[0]
    indecies_shuffled=copy.copy(neg_indxs)
    np.random.shuffle(indecies_shuffled)
    num_subsampled = size*imbalance_ratio
    if num_subsampled > len(neg_indxs):
        print '    Error: Not enough to subsample'
        exit
    else:
        new_neg_indxs = indecies_shuffled[:num_subsampled]
    indx = np.concatenate((pos_indxs, new_neg_indxs))
    return y[indx], y_pred[indx]   

def binarize_w_unlabeled(matrix, thres):
    matrix2=copy.copy(matrix)
    matrix2[matrix == -1] = -1
    matrix2[matrix > thres] = 1
    matrix2[np.logical_and(matrix>=0, matrix<thres)] = 0
    return matrix2

def get_kernel_pairs_distance_matched(labels, preds,  pair_dist, min_dist, max_dist, dist_step, imbalance_ratio, thres):

    def subsample_indx(indecies, size, imbalance_ratio):
        indecies_shuffled=copy.copy(indecies)
        np.random.shuffle(indecies_shuffled)
        num_subsampled = size*imbalance_ratio
        if num_subsampled > len(indecies[0]):
            print '    Error: Not enough to subsample'
            exit
        #print indecies_shuffled[0].shape
        #print indecies_shuffled[0][:num_subsampled].shape
        else:
            return indecies_shuffled[0][:num_subsampled]

    neg_indxs = np.where(labels.astype(float)<=thres)[0]
    pos_indxs = np.where(labels.astype(float)>thres)[0]
    labels_pos=labels[pos_indxs]
    labels_neg=labels[neg_indxs]
    preds_pos=preds[pos_indxs]
    preds_neg=preds[neg_indxs]
    pair_dist_pos=pair_dist[pos_indxs]
    pair_dist_neg=pair_dist[neg_indxs]

    thres1=min_dist+dist_step
    thres2=min_dist

    labels_new=np.empty(([0,]))
    preds_new=np.empty(([0,]))
    pair_dist_new=np.empty(([0,]))

    while thres1 <= max_dist:
        print 'distance window: ', '[', thres2, ',', thres1, ']'
        neg_indx_at_dist=np.where((pair_dist_neg[:,].astype(int) <= thres1) & (pair_dist_neg[:,].astype(int) > thres2))
        pos_indx_at_dist=np.where((pair_dist_pos[:,].astype(int) <= thres1) & (pair_dist_pos[:,].astype(int) > thres2))
        if len(pos_indx_at_dist[0])> len(neg_indx_at_dist[0]):
            print 'more pos than neg'
            print 'number of pos at distance=',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(pos_indx_at_dist, len(neg_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=indx_subsampled
            new_neg_indx_at_dist=neg_indx_at_dist[0]

        else:
            #print 'more neg than pos'
            print 'number of pos at distance=',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(neg_indx_at_dist, len(pos_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=pos_indx_at_dist[0]
            new_neg_indx_at_dist=indx_subsampled

        labels_pos_at_dist=labels_pos[new_pos_indx_at_dist]
        labels_neg_at_dist=labels_neg[new_neg_indx_at_dist]
        preds_pos_at_dist=preds_pos[new_pos_indx_at_dist]
        preds_neg_at_dist=preds_neg[new_neg_indx_at_dist]
        #pair_dist_pos_at_dist=pair_dist_pos[new_pos_indx_at_dist]
        #pair_dist_neg_at_dist=pair_dist_neg[new_neg_indx_at_dist]

        labels_at_dist=np.concatenate((labels_pos_at_dist, labels_neg_at_dist))
        preds_at_dist=np.concatenate((preds_pos_at_dist, preds_neg_at_dist))
        #pair_dist_at_dist=np.concatenate((pair_dist_pos_at_dist, pair_dist_neg_at_dist))

        print 'labels at dist: ', labels_at_dist.shape
        print 'preds at dist: ', preds_at_dist.shape
        #print 'pair_dist at dist: ', pair_dist_at_dist.shape

        labels_new=np.concatenate((labels_new, labels_at_dist))
        preds_new=np.concatenate((preds_new, preds_at_dist))
        #pair_dist_new=np.concatenate((pair_dist_new, pair_dist_at_dist))


        #print X_new.shape, X_at_dist.shape
        #print y_new.shape, y_at_dist.shape
        #print indx_new.shape, indx_at_dist.shape

        #print "# of neg:", np.where(y_at_dist==0)[0].shape
        #print "# of pos:", np.where(y_at_dist==1)[0].shape
        #thres2=thres1+min_dist
        thres2=thres1
        thres1=thres1+dist_step

    return np.expand_dims(binarize(labels_new, thres=thres).astype(int), axis=1), np.expand_dims(preds_new, axis=1)



