# import packages
from __future__ import division
import networkx as nx
import numpy as np
import os
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import itertools

from numpy.linalg import inv

import csv

import parsing as parse_mat

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## function to create + save dictionary of features
def create_dict(key, obj):
    return(dict([(key[i], obj[i]) for i in range(len(key))]))


# set the working directory and import helper functions
#get the current working directory and then redirect into the functions under code 
cwd = os.getcwd()
#print("cwd", cwd)
# import the data from the data folder
data_cwd = cwd+ '/data'
#print("data_cwd ", data_cwd )

# change the working directory and import the fb dataset
fb100_file = data_cwd +'/Amherst41'



A, metadata = parse_mat.parse_fb100_mat_file(fb100_file)

# change A(scipy csc matrix) into a numpy matrix
adj_matrix_tmp = A.todense()
#get the gender for each node(1/2,0 for missing)
gender_y_tmp = metadata[:,1] 
# get the corresponding gender for each node in a disctionary form
gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)

(graph, gender_y)  = parse_mat.create_graph(adj_matrix_tmp,gender_dict,'gender',0,None,'yes')


percent_initially_labelled = [0.01, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


random_seed_list = [1, 12345, 837, 2841, 4293, 6305, 6746, 9056, 9241, 9547]



def get_train_test_10_split(data_cwd, initially_labeled_precent,random_seed):
    # first open test file
    test_index_file_name = "gender_test_indicies_"+ "{0:0>4}".format(random_seed) + "rand_"+str(initially_labeled_precent)+ "pct.txt"
    train_index_file_name = "gender_train_indicies_"+"{0:0>4}".format(random_seed) + "rand_"+str(initially_labeled_precent)+ "pct.txt"

    # read and convert the text_index.txt into an numpy array
    with open(test_index_file_name) as f:
        test_index_file = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    test_index = np.array([int(x.strip()) for x in test_index_file] )

    with open(train_index_file_name) as f:
        train_index_file = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    train_index = np.array([int(x.strip()) for x in train_index_file])

    return train_index, test_index


def ds_prepare(graph):
    # get W matrix, Z(for row sum) and Z_prime(for col sum), and A_tilde
    # get matrix W:
    A = np.array(nx.adjacency_matrix(graph).todense()) 
    d = np.sum(A, axis=1)
    D = np.diag(d)

    sigma_square = 0.1

    # Alternative way(19): set Sigma_square = sigma_square./d, where sigma_square is fixed
    Sigma_square = np.divide(sigma_square,d)

    Sigma = np.diag(Sigma_square)
    W = np.dot(A,inv(Sigma))
    w_col_sum = np.sum(W, axis=0)
    w_row_sum = np.sum(W, axis=1)
    Z_prime = np.diag(w_col_sum)
    Z = np.diag(w_row_sum)
    A_tilde = np.dot(np.dot(W,inv(Z_prime)),np.transpose(W))
    return (A_tilde)

def ZGL(adj_matrix_gender,gender_y,percent_initially_labelled, num_iter):
    W_unordered = np.array(adj_matrix_gender)

#    percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)    
    mean_accuracy = []
    se_accuracy = []

    mean_micro_auc = []
    se_micro_auc = []

    mean_wt_auc = []
    se_wt_auc = []

   # auprc 
    mean_auprc = []
    se_auprc = []
    
    # f1
    mean_f1 = []
    se_f1 = []

    n = len(gender_y)
    # see how many classes are there and rearrange them
    classes = np.sort(np.unique(gender_y))
    class_labels = np.array(range(len(classes)))

    # relabel membership class labels - for coding convenience
    # preserve ordering of original class labels -- but force to be in sequential order now
    gender_y_update = np.copy(gender_y)
    for j in range(len(classes)):
        gender_y_update[gender_y_update == classes[j]] = class_labels[j]
        
    cwd = os.getcwd()
    print("ds data dir", cwd)
        
    data_cwd = cwd + '/index'
    os.chdir(data_cwd)    
    print(data_cwd)

    for i in range(len(percent_initially_labelled)):
        print(percent_initially_labelled[i]) 
    
        accuracy = [] 
        micro_auc = []
        wt_auc = []
        # auprc
        auprc = []
        # f1 score
        f1 = []

        for j in range(len(random_seed_list)):
            #for train_index, test_index in k_fold.split(W_unordered, gender_y_update):
            train_index, test_index = get_train_test_10_split(data_cwd, percent_initially_labelled[i], random_seed_list[j])
            X_train, X_test = W_unordered[train_index], W_unordered[test_index]
            y_train, y_test = gender_y_update[train_index], gender_y_update[test_index]
            #print(train_index)
            idx = np.concatenate((train_index, test_index)) # concatenate train + test = L + U
            # rearrange the column of W matrix to be train + test = L + U 
            W = np.reshape([W_unordered[row,col] for row in np.array(idx) for col in np.array(idx)],(n,n))    

            #fl: L*c(size) label matrix from ZGL paper
            train_labels = np.array([np.array(gender_y_update)[id] for id in train_index]) # resort labels to be in same order as training data
            classes_train = np.sort(np.unique(train_labels))
            ##get the approximate ratio of the max class labels
            accuracy_score_benchmark = np.mean(np.array(train_labels) == np.max(class_labels))
            # get the fl label vector from ZGL paper
            fl =np.array(np.matrix(label_binarize(train_labels,list(classes_train) + [np.max(classes_train)+1]))[:,0:(np.max(classes_train)+1)]) 
            # record testing gender labels for comparing predictions -- ie ground-truth
            true_test_labels = np.array([np.array(gender_y_update)[id] for id in test_index])
            classes_true_test = np.sort(np.unique(true_test_labels))
            ground_truth =np.array(np.matrix(label_binarize(true_test_labels,list(classes_true_test) + [np.max(classes_true_test)+1]))[:,0:(np.max(classes_true_test)+1)])

            l = len(train_index) # number of labeled points
            u = len(test_index) # number of unlabeled points

            ## compute Equation (5) in ZGL paper
            W_ll = W[0:l,0:l]
            W_lu = W[0:l,l:(l+u)]
            W_ul = W[l:(l+u),0:l]
            W_uu = W[l:(l+u),l:(l+u)]
            # get the D matrix(numpy/scipy are different)

            D = np.diag(np.sum(W, axis=1))
            D_ll = D[0:l,0:l]
            D_lu = D[0:l,l:(l+u)]
            D_ul = D[l:(l+u),0:l]
            D_uu = D[l:(l+u),l:(l+u)]
            # harmonic_fxn is just fu
            harmonic_fxn =  np.dot(np.dot(np.linalg.inv(np.matrix(np.subtract(D_uu, W_uu))),np.matrix(W_ul)), np.matrix(fl))
            # if the classifications are greater than 2
            if len(np.unique(gender_y_update))>2:
                micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn,average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn,average='weighted'))
                accuracy.append(metrics.accuracy_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn))
                # add auprc
                auprc.append(metrics.average_precision_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn,average='micro'))
            # if there are only two types
            else:
                micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),
                                                       harmonic_fxn[:,1]-harmonic_fxn[:,0],average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),
                                                    harmonic_fxn[:,1]-harmonic_fxn[:,0],average='weighted'))

                auprc.append(metrics.average_precision_score(label_binarize(gender_y[test_index],np.unique(gender_y)),
                                                       harmonic_fxn[:,1]-harmonic_fxn[:,0],average='micro')) 
                y_true = label_binarize(gender_y[test_index],np.unique(gender_y))
                y_pred = ((harmonic_fxn[:,1]) > accuracy_score_benchmark)+0

                f1.append(metrics.f1_score(y_true, y_pred,average='macro', sample_weight=None))
                accuracy.append(metrics.accuracy_score(y_true, y_pred))
            
        # get the mean and standard deviation
        mean_accuracy.append(np.mean(accuracy))  
        se_accuracy.append(np.std(accuracy)) 
        
        mean_micro_auc.append(np.mean(micro_auc))
        se_micro_auc.append(np.std(micro_auc))
        mean_wt_auc.append(np.mean(wt_auc))
        se_wt_auc.append(np.std(wt_auc))
    
        # auprc
        mean_auprc.append(np.mean(auprc))
        se_auprc.append(np.std(auprc))
        
        # f1 score
        mean_f1.append(np.mean(f1))
        se_f1.append(np.std(f1))
           
        
    return(mean_accuracy,se_accuracy,mean_micro_auc,se_micro_auc,mean_wt_auc,se_wt_auc, mean_f1, se_f1,mean_auprc, se_auprc)


n_iter = 10
(graph, gender_y)  = parse_mat.create_graph(adj_matrix_tmp,gender_dict,'gender',0,None,'yes')
A_tilde = ds_prepare(graph)

(mean_accuracy_baselineDS,se_accuracy_baselineDS,mean_micro_auc_baselineDS,se_micro_auc_baselineDS,mean_wt_auc_baselineDS,se_wt_auc_baselineDS,mean_f1_baselineDS,se_f1_baselineDS,mean_auprc_baselineDS,se_auprc_baselineDS) = ZGL(np.array(A_tilde),gender_y,percent_initially_labelled, n_iter)

# write the baseline result into an csv file named baseline_results.csv
cwd = os.getcwd()
# set the output directory: cwd is /index, get the parents dir
parent_cwd = os.path.dirname(cwd)

# field names  
col_names = ['Type', 'Labeled Percent', 'metric', '1%', '5%','10%','20%','30%','40%','50%','60%','70%','80%','90%'] 

filename = parent_cwd + "/baseline_result.csv"

# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(col_names)  
        
    # writing the data rows  
    row1 = ["mean", "DS ORIG", "auroc"] + [str(i) for i in mean_wt_auc_baselineDS]
    csvwriter.writerow(row1)
    row2 = ["mean", "DS ORIG", "auroc"] + [str(i) for i in se_wt_auc_baselineDS]
    csvwriter.writerow(row2)
    row3 = ["mean", "DS ORIG", "auroc"] + [str(i) for i in mean_accuracy_baselineDS]
    csvwriter.writerow(row3)
    row4 = ["mean", "DS ORIG", "auroc"] + [str(i) for i in se_accuracy_baselineDS]
    csvwriter.writerow(row4)










