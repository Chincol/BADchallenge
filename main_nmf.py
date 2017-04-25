'''
SUMMARY:  bird detection, joint classification and classification trained DNN
          acc of 83% will be obtained on testing set after 1000 iterations. 
AUTHOR:   Qiuqiang Kong
Created:  2016.11.23
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/cvsspstf/is0017/work/AED')
sys.path.append('/user/cvsspstf/is0017/work')
import numpy as np
np.random.seed(1515)
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import seaborn as sns
import prepare_data as pp_data
import config as cfg
import cPickle
import time
from nmf import NMF
import librosa
from gap_vbem import * 
from sklearn import decomposition, ensemble, metrics, preprocessing
from joblib import Parallel, delayed
import multiprocessing
import nmf_function 
import spherical_kmeans 
import ipdb


tr_fe_fd_d1 = cfg.wbl_mel_fd        # training feature folder
te_fe_fd_d1 = cfg.wbl_mel_fd        # testing feature folder
tr_fe_fd_d2 = cfg.ff_mel_fd       # training feature folder
te_fe_fd_d2 = cfg.ff_mel_fd       # testing feature folder

# tr_fe_fd_d1 = cfg.wbl_fft_fd        # training feature folder
# te_fe_fd_d1 = cfg.wbl_fft_fd        # testing feature folder
# tr_fe_fd_d2 = cfg.ff_fft_fd       # training feature folder
# te_fe_fd_d2 = cfg.ff_fft_fd       # testi
tr_cv_csv_path_d1 = cfg.wbl_cv10_csv_path     # training cv csv file
te_cv_csv_path_d1 = cfg.wbl_cv10_csv_path     # testing cv csv file
tr_cv_csv_path_d2 = cfg.ff_cv10_csv_path      # training cv csv file
te_cv_csv_path_d2 = cfg.ff_cv10_csv_path      # testing cv csv file
tr_fold = [0]   # training folds, should be list of nums, e.g. [0,2,4]
te_fold = [1]   # testing folds, should be list of nums, e.g. [0,2,4]

rank_p=int(sys.argv[2])
rank_n=int(sys.argv[3])
type=sys.argv[4]
# W_name='W/W_kmeans_9folds.p'
n_trees=500
sh_order = int(sys.argv[5])

W_name='W/W_mel_'+type+'_kl_'+str(rank_p)+'p_'+str(rank_n)+'n_sh'+str(sh_order)+'.p'
results_name= 'results/W_mel_'+type+'_kl_'+str(rank_p)+'p_'+str(rank_n)+'n_sh'+str(sh_order)+'.p'

num_cores = multiprocessing.cpu_count()
eps = np.spacing(1)

### Pre load all feature files to a pickle file to speed up. 
def pre_load():
    # tr_X.shape: (n_songs, n_chunks, n_freq), tr_mask.shape: (n_songs, n_chunks)
    # tr_y.shape: (n_songs, n_out), len(tr_na_list): n_songs
    tr_X_d1, tr_y_d1, tr_na_list_d1 = pp_data.GetMiniData3dSongWise( tr_fe_fd_d1, tr_cv_csv_path_d1, tr_fold )
    tr_X_d2, tr_y_d2, tr_na_list_d2 = pp_data.GetMiniData3dSongWise( tr_fe_fd_d2, tr_cv_csv_path_d2, tr_fold )       
    te_X_d1, te_y_d1, te_na_list_d1 = pp_data.GetMiniData3dSongWise( te_fe_fd_d1, te_cv_csv_path_d1, te_fold )
    te_X_d2, te_y_d2, te_na_list_d2 = pp_data.GetMiniData3dSongWise( te_fe_fd_d2, te_cv_csv_path_d2, te_fold )
    
    tr_X=tr_X_d1+tr_X_d2
    tr_y=np.hstack((tr_y_d1,tr_y_d2))
    tr_na_list=tr_na_list_d1+tr_na_list_d2
    te_X=te_X_d1+te_X_d2
    te_y=np.hstack((te_y_d1,te_y_d2))
    te_na_list=te_na_list_d1+te_na_list_d2
    
    dict = {}
    dict['tr_X'],  dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_y'], dict['te_na_list'] = tr_X,  tr_y, tr_na_list, te_X,  te_y, te_na_list
    
    cPickle.dump( dict, open( 'pre_load.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    print "Pre load finished!"


### train the model
def train():
    
    # ----------- Load data -----------
    # load data
    dict = cPickle.load( open( 'pre_load.p', 'rb' ) )
    tr_X, tr_y, tr_na_list, te_X,  te_y, te_na_list = dict['tr_X'], dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_y'], dict['te_na_list']

    tr_positive = np.take(tr_X,np.where(tr_y==1)[0])
    tr_positive=[ t/np.max(t) for t in tr_positive]
    tr_positive=[librosa.feature.stack_memory(t.transpose(), n_steps=sh_order) for t in tr_positive]
    tr_negative = np.take(tr_X,np.where(tr_y==0)[0])
    tr_negative=[ t/np.max(t) for t in tr_negative]
    tr_negative=[librosa.feature.stack_memory(t.transpose(), n_steps=sh_order) for t in tr_negative]
    
    # # # ----------- Do training seperate bases for each file-----------
    # nmf_model=NMF(rank_p, norm_W=1,  iterations=500, update_func = "kl", verbose=True)
    # W_positive=[]
    # for f in tr_positive:
    #     [W,H,error]=nmf_model.process(f.transpose())
    #     W_positive.append(W)
    # 
    # nmf_model=NMF(rank_p, norm_W=1,  iterations=500, update_func = "kl", verbose=True)
    # W_negative=[]
    # for f in tr_negative:
    #     [W,H,error]=nmf_model.process(f.transpose())
    #     W_negative.append(W)
    
    tr_positive = np.hstack(tr_positive)
    tr_negative = np.hstack(tr_negative)
    train_data=np.hstack((tr_positive,tr_negative))
    print >> sys.stderr, train_data.shape
    # 
    # # # ----------- Do training overcomplete dictionary -----------
    # p = decomposition.PCA(whiten=True, n_components= 0.99)
    # pca_data=p.fit_transform(train_data)
   
#   #   num=500
    # num_dim=pca_data.shape[1]
    # num_training_samples=pca_data.shape[0]
    # km = spherical_kmeans.OSKmeans(num,num_dim)
    # print "Learning k-means: "+ str(num)
    # for _ in range(1000):
    #    print _
    #    for index in range(num_training_samples):
    #        km.update(pca_data[index,:])
    # codebook=km.centroids
    # cPickle.dump( [codebook, p], open( W_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )

    # # # ----------- Do training -----------
    if type == '0_1':
        print >> sys.stderr, "NMF on positive examples"
        nmf_model=NMF(rank_p, norm_W=1,  iterations=200, update_func = "kl", verbose=False)
        [W_positive,H,error]=nmf_model.process(tr_positive)
        # # # a_H=np.ones(rank_n+rank_p)
        # # # b_H=np.ones(rank_n+rank_p) 
        # # # [error, W_positive, H_gap] = gap_vbem(tr_positive, rank_n+rank_p, a_H, b_H, iterations=100, verbose=True)
        # # 
        print >> sys.stderr, "NMF on negative examples"
        nmf_model=NMF(rank_n, norm_W=1,  iterations=200, update_func = "kl", verbose=False)
        [W_negative,H,error]=nmf_model.process(tr_negative)
        # # # [error, W_negative, H_gap] = gap_vbem(tr_negative, rank_n+rank_p, a_H, b_H, iterations=100, verbose=True)
        cPickle.dump( [W_positive, W_negative], open( W_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    elif type == 'unsupervised':
        print >> sys.stderr, "unsupervised NMF"
        V=np.hstack((tr_negative,tr_positive))
        nmf_model=NMF(rank_n+rank_p, norm_W=1,  iterations=200, update_func = "kl", verbose=False)
        [W,H,error]=nmf_model.process(V) 
        print >> sys.stderr, error
        # # a_H=np.ones(rank_n+rankwork/bird_backup/W/W_mel_01_kl_50p_50_9folds.n.p_p)
        # # b_H=np.ones(rank_n+rank_p)   
        # # [error, W_gap, H_gap] = gap_vbem(V, rank_n+rank_p, a_H, b_H, H0, iterations=100, verbose=False)
        #    
        cPickle.dump( W, open( W_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    elif type == '01':
        # # -------- Train with masking ----------
        print >> sys.stderr, "masked NMF on training files"
        mask=np.zeros((rank_p, tr_negative.shape[1]))
        V=np.hstack((tr_negative,tr_positive))
        H0=np.random.rand(rank_n+rank_p, V.shape[1])+eps
        H0[-mask.shape[0]:,:mask.shape[1]]=mask
        nmf_model=NMF(rank_n+rank_p, norm_W=1,  iterations=200, update_func = "kl", verbose=False)
        [W,H,error]=nmf_model.process(V,H0=H0) 
        print >> sys.stderr, error
        # # a_H=np.ones(rank_n+rankwork/bird_backup/W/W_mel_01_kl_50p_50_9folds.n.p_p)
        # # b_H=np.ones(rank_n+rank_p)   
        # # [error, W_gap, H_gap] = gap_vbem(V, rank_n+rank_p, a_H, b_H, H0, iterations=100, verbose=False)
        #    
        cPickle.dump( W, open( W_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    else:
        raise ValueError('Dictionary type not recognized')
    
    print >> sys.stderr, "Dictionary " +W_name +" finished!"
    
    
### recognize
def classify():
    # load model
    print "Loading dictionary "+ W_name
    if type == '01' or type =='unsupervised':
        W = cPickle.load( open( W_name, 'rb' ) )
    elif type == '0_1':
        [W_positive,W_negative] = cPickle.load( open( W_name, 'rb' ) )
        W=np.hstack((np.vstack(W_positive),np.vstack(W_negative)))
    else:
        raise ValueError('Dictionary type not recognized')
    # codebook = cPickle.load( open( W_name, 'rb' ) )
    
    # load data
    dict = cPickle.load( open( 'pre_load.p', 'rb' ) )
    tr_X,  tr_y, tr_na_list, te_X,  te_y, te_na_list = dict['tr_X'],  dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_y'], dict['te_na_list']
    n_songs = len(te_X)

    # tr_positive = np.take(tr_X,np.where(tr_y==1)[0])
    # tr_positive=[ t/np.max(t) for t in tr_positive]
    # tr_negative = np.take(tr_X,np.where(tr_y==0)[0])
    # tr_negative=[ t/np.max(t) for t in tr_negative]
    #  
    # tr_positive = np.vstack(tr_positive)
    # tr_negative = np.vstack(tr_negative)
    # train_data=np.vstack((tr_positive,tr_negative))

    
    print >> sys.stderr, "NMF of training files"
    f_norm = [f/(np.sqrt(np.sum(np.array(f)**2,axis=0))) for f in tr_X]
    f_norm=[librosa.feature.stack_memory(t.transpose(), n_steps=sh_order) for t in f_norm]
    train_list = Parallel(n_jobs=num_cores)(delayed(nmf_function.process)(W.shape[1], f, W0 = W,  iterations=100) for f in f_norm)
    # mean and standard deviation
    data_pooled =[(np.hstack((np.mean(sample[1], axis=1), np.std(sample[1], axis=1)))) for sample in train_list]
    
    # max pooling
    # data_pooled =[np.max(sample[1],axis=1) for sample in train_list]
    # data_pooled =[np.hstack(sample[1][:,:150]) for sample in train_list]
    # print "K-means coding of training files"
    # f_norm = [f/(np.sqrt(np.sum(np.array(f)**2,axis=0))) for f in tr_X]
    # p = decomposition.PCA(whiten=True, n_components= 0.99)
    # pca_data=p.fit_transform(train_structured dropout for weak label and multipledata)
    # train_list= [np.dot(p.transform(sample), codebook.transpose()) for sample in f_norm]     # linear encoding scheme
    # # data_pooled =[(np.hstack((np.mean(sample[:,1]), np.std(sample[:,1])))) for sample in train_list]
    # data_pooled =[np.max(sample[1]) for sample in train_list]
    print >> sys.stderr, "Training Random Forest"
    clf =ensemble.RandomForestClassifier(n_trees, n_jobs=-1)
    clf.fit(np.array(data_pooled), np.array(tr_y))
    
    print >> sys.stderr, "NMF of test files"
    f_norm = [f/(np.sqrt(np.sum(np.array(f)**2,axis=0))) for f in te_X]  
    f_norm=[librosa.feature.stack_memory(t.transpose(), n_steps=sh_order) for t in f_norm]  
    test_list = Parallel(n_jobs=num_cores)(delayed(nmf_function.process)(W.shape[1], f, W0 = W,  iterations=100) for f in f_norm)
    # mean and standard deviation pooling
    test_data_pooled =[(np.hstack((np.mean(sample[1], axis=1), np.std(sample[1], axis=1)))) for sample in test_list]
    # max pooling
    # test_data_pooled =[np.max(sample[1],axis=1) for sample in test_list]
    # test_data_pooled =[np.hstack(sample[1][:,:150]) for sample in test_list]
    # print "k-means encoding of test files"
    # test_list= [np.dot(p.transform(sample), codebook.transpose()) for sample in f_norm]     # linear encoding scheme
    # # test_data_pooled =[(np.hstack((np.mean(sample[:,1]), np.std(sample[:,1])))) for sample in test_list]
    # 
    # test_data_pooled =[np.max(sample[1]) for sample in test_list]
    print >> sys.stderr ,"Predicting with Random Forest"
    y_scores=clf.predict_proba(np.array(test_data_pooled))
    cPickle.dump( y_scores, open( results_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    fpr, tpr, thresholds = metrics.roc_curve(te_y, y_scores[:,1])
    
    roc_auc = metrics.auc(fpr, tpr)
    
    ion()
    plt.plot(fpr,tpr,label=W_name)
    plt.legend(loc='best')
    print roc_auc
    
    # confM = np.zeros((2,2))
    # for i1 in xrange(n_songs):
    #     confM[ te_y[i1], predicted_classes[i1] ] += 1

    # print confM
    # print 'acc:', np.sum(np.diag(confM)) / np.sum(confM)    

### classify spectrograms directly
def classify_spec():
    # load data
    dict = cPickle.load( open( 'pre_load.p', 'rb' ) )
    tr_X,  tr_y, tr_na_list, te_X,  te_y, te_na_list = dict['tr_X'],  dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_y'], dict['te_na_list']
    n_songs = len(te_X)

    f_norm = [f/(np.sqrt(np.sum(np.array(f)**2,axis=0))) for f in tr_X]
    data=[ np.hstack(np.vstack(sample[:118])) for sample in f_norm]
    data_pooled_mean =[(np.hstack((np.mean(sample, axis=0), np.std(sample, axis=0)))) for sample in f_norm]
    data_pooled_max =[np.max(sample, axis=0) for sample in f_norm]
    
    print "Training Random Forest"
    clf1 =ensemble.RandomForestClassifier(n_trees, n_jobs=-1)
    clf1.fit(np.array(data), np.array(tr_y))
    
    clf2 =ensemble.RandomForestClassifier(n_trees, n_jobs=-1)
    clf2.fit(np.array(data_pooled_mean), np.array(tr_y))
    
    clf3 =ensemble.RandomForestClassifier(n_trees, n_jobs=-1)
    clf3.fit(np.array(data_pooled_max), np.array(tr_y))
    
    f_norm = [f/(np.sqrt(np.sum(np.array(f)**2,axis=0))) for f in te_X]    
    test_data=[ np.hstack(np.vstack(sample[:118])) for sample in f_norm]
    test_data_pooled_mean =[(np.hstack((np.mean(sample, axis=0), np.std(sample, axis=0)))) for sample in f_norm]
    test_data_pooled_max =[np.max(sample, axis=0) for sample in f_norm]

    print "Predicting with Random Forest"
    y_scores1=clf1.predict_proba(np.array(test_data))
    fpr1, tpr1, thresholds = metrics.roc_curve(te_y, y_scores1[:,1])
    roc_auc1 = metrics.auc(fpr1, tpr1)
   
    y_scores2=clf2.predict_proba(np.array(test_data_pooled_mean))
    fpr2, tpr2, thresholds = metrics.roc_curve(te_y, y_scores2[:,1])
    roc_auc2 = metrics.auc(fpr2, tpr2)
    
    y_scores3=clf3.predict_proba(np.array(test_data_pooled_max))
    fpr3, tpr3, thresholds = metrics.roc_curve(te_y, y_scores3[:,1])
    roc_auc3 = metrics.auc(fpr3, tpr3)
    
    print 'acc: ', roc_auc1, roc_auc2, roc_auc3
    plt.plot(fpr1,tpr1, label='full_mel')
    plt.plot(fpr2,tpr2, label='mean')
    plt.plot(fpr3,tpr3, label='max')
    plt.legend(loc='best')
    # confM = np.zeros((2,2))
    # for i1 in xrange(n_songs):
    #     confM[ te_y[i1], predicted_classes[i1] ] += 1

    #  print confM
    # print 'acc:', np.sum(np.diag(confM)) / np.sum(confM)    

### main function
if __name__ == '__main__':
    assert len( sys.argv )==6, "\nUsage: \npython main_nmf.py --train\npython main_dev_dnn.py --recognize"
    if sys.argv[1] == '--pre_load': pre_load()
    if sys.argv[1] == '--train': train()
    if sys.argv[1] == '--classify': classify()
    if sys.argv[1] == '--classify_spec': classify_spec()
