import numpy as np
import pandas as pd
import copy
import comonotonic as cm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import math
import operator
from scipy.stats import norm

# given a list containing text values in a column
# encode to 0, 1, 2...
def encoding(column_val):
    encountered = {} # {text value: encoder}
    encoded = []
    encoder = 0
    for val in column_val:
        if val not in encountered.keys():
            encountered[val] = encoder
            encoder += 1
        encoded.append(encountered[val])
    return encoded

def encode_df(df, encoded_columns):
    df_copy = df.copy()
    for col in encoded_columns:
        encoded = encoding(df.iloc[:,col])
        if col == df.shape[1] - 1: # Y needs to be encoded
            df_copy['Y'] = encoded
        else:
            df_copy['X'+str(col)] = encoded
    return df_copy

def normalize_df_col(df, cont_columns):
    for col in cont_columns:
        colname = 'X' + str(col)
        col_val = df[[colname]].values
        col_val = list(col_val.ravel())
        mean = np.mean(col_val)
        std = np.std(col_val)
        col_scaled = list(map(lambda x:(x-mean)/std, col_val))
        df[colname] = col_scaled

# split the dataframe, discretize the training set and get the bins, discretize the test set based on the bins
def df_split_discretize(df, random_state, qcut, allocation_book):
    X = df.iloc[:,0:-1]
    Y = df.iloc[:,-1]
    for col_idx in allocation_book.keys():
        if qcut == False:    
            discretized_col = pd.cut(X.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
            discretized_col = discretized_col.astype('int32')
        else:
            try:
                discretized_col = pd.qcut(X.iloc[:,col_idx],allocation_book[col_idx], 
                                        labels=[i for i in range(allocation_book[col_idx])])
            except:
                discretized_col = pd.cut(X.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
            discretized_col = discretized_col.astype('int32')
        X['X'+str(col_idx)] = discretized_col
    X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(X,Y,test_size=0.2,random_state=random_state)
    return X_train_df, X_test_df, Y_train_df, Y_test_df

def merge_dict(d1, d2):
    d = {}
    for k1 in d1.keys():
        d[k1] = d1[k1]
    for k2 in d2.keys():
        d[k2] = d2[k2]
    return d

def get_accuracy(y_predict, y_test):
    t = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
            t += 1        
    return t/len(y_test)

######## for clustering by correlation coefficient matrix
# c1, c2 are two lists, compute the distance between them
def get_distance(distance_matrix, c1, c2):
    min_dist = None
    for f1 in c1:
        for f2 in c2:
            dist = distance_matrix[f1][f2]
            if min_dist == None or dist < min_dist:
                min_dist = dist
    return min_dist

# update cluster_book
def update_cluster_book(distance_matrix, cluster_book):
    min_dist = None
    merger1 = None
    merger2 = None
    for c1 in cluster_book:
        for c2 in cluster_book:
            if c1 != c2:
                dist = get_distance(distance_matrix, c1, c2)
                if min_dist == None or min_dist > dist:
                    merger1 = c1.copy()
                    merger2 = c2.copy()
                    min_dist = dist
    if merger1 == None or merger2 == None:
        return cluster_book, 1
    cluster_book.remove(merger1)
    cluster_book.remove(merger2)
    cluster_book.append(merger1+merger2)
    return cluster_book, min_dist

# stopping condition: the current minimum distance among clusters larger than max_distance
def cluster_agnes(distance_matrix, max_distance):
    cluster_book = [[i] for i in range(len(distance_matrix))] # list of list, store the current clusters
    cluster_book_copy = cluster_book.copy() # in case clustering is unnecessary
    cluster_book, min_dist = update_cluster_book(distance_matrix, cluster_book) 
    cluster_necessary = False # if clustering is unnecessary, just use cluster_book_copy
    while min_dist <= max_distance:
        cluster_necessary = True
        cluster_book, min_dist = update_cluster_book(distance_matrix, cluster_book)
    if cluster_necessary == False:
        return cluster_book_copy
    else:
        return cluster_book
    

# use weighted average of naive bayes and pure comonotonic
def weighted_avg(nb_dist, p_como_dist, w_nb):
    prob_dist = {c: (nb_dist[c]*w_nb+p_como_dist[c]*(1-w_nb)) for c in nb_dist.keys()}
    return max(prob_dist.items(), key=operator.itemgetter(1))[0]

# scale the probability density to magnitude 10^(-1)
def scaler(prob_density):
    i = 0
    while prob_density < 0.1:
        prob_density *= 10
        i += 1
    return i, prob_density

# pass in a numpy 1d array of samples of a continuous variable
def auto_discretize(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    bins = np.array([mean-3*std, mean-2*std, mean-std, mean, mean+std, mean+2*std, mean+3*std])
    discretized = np.digitize(feature, bins)
    return discretized, bins

def custom_discretize(feature, num_classes):
    sup = max(feature)
    inf = min(feature)
    stride = (sup - inf)/num_classes
    bins = [inf+stride*(i+1) for i in range(num_classes-1)]
    discretized = np.digitize(feature, bins)
    return discretized, bins

def ecdf(train_x, val):
    return len([i for i in train_x if i <= val])/len(train_x)

def copula_func(u, R):
    # note that u is empirical cdf
    vec = np.array([norm.ppf(i) for i in u])
    a = 1/np.sqrt(abs(np.linalg.det(R)))
    if np.linalg.det(R) == 0:
        for i in range(len(R)):
            for j in range(len(R[i])):
                if i != j:
                    R[i][j] += 0.01
    mid_mat = np.linalg.inv(R) - np.identity(len(u), dtype=float)
    b = np.matmul(vec, mid_mat)
    c = np.matmul(b, vec.T)
    return a * np.exp(-0.5 * c)