import numpy as np
import pandas as pd
import copy
import comonotonic as cm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import math
import operator

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

# to test the accuracy when a certain allocation of number of categories is employed 
def accuracy_test(allocation_book, df, unrankable_fv, colnames, min_corr, use_cluster, random_state, qcut): 
    # allocation book is ONLY for cont faetures {feature_idx: num_of_categories}; 
    # unrankable_fv is a dict {unrankable_feature: fv's}
    # df is the encoded but NOT categorized. DO NOT CHANGE df!!!
    if unrankable_fv != None:
        unrankable = list(unrankable_fv.keys())
        feature_val = merge_dict(unrankable_fv, allocation_book)
    else:
        unrankable = None
        feature_val = allocation_book.copy()
    # deal with discrete rankable
    for col in range(df.shape[1]-1):
        if col not in feature_val.keys():
            feature_val[col] = df['X'+str(col)].nunique()
    if use_cluster == True:
        uncategorized_df = df.copy()
    X_train_df, X_test_df, Y_train_df, Y_test_df = df_split_discretize(df, random_state, qcut, allocation_book)
    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()
    Y_train = Y_train_df.to_numpy()
    Y_test = Y_test_df.to_numpy()
    if use_cluster == False: # use pure comonotonic
        pure_como_classifier = cm.pure_comonotonic(X_train, Y_train, unrankable = unrankable, feature_val = feature_val)
        pure_como_classifier.run()
        Y_predict = pure_como_classifier.predict(X_test)
    else: # use clustered comonotonic
        #uncategorized_x = uncategorized_df[colnames[:-1]].to_numpy()
        cluster_como_classifier = cm.clustered_comonotonic(X_train, Y_train, unrankable = unrankable, feature_val = feature_val,
                                                        uncategorized_df = uncategorized_df, colnames = colnames,
                                                        min_corr = min_corr, random_state = random_state)
        cluster_como_classifier.run_cluster()
        Y_predict = cluster_como_classifier.predict(X_test)
    accuracy = get_accuracy(Y_predict, Y_test)
    return accuracy


"""
This is actually a mistaken accuracy test because it changes the value of df after the accuracy test
However, after iterations the accuracy of pure comonotonic rises up for adult.csv dataset
"""
def mistaken_accuracy_test(allocation_book, df, unrankable_fv, colnames, min_corr, use_cluster, random_state, qcut): 
    # allocation book is ONLY for cont features {feature_idx: num_of_categories}; 
    # unrankable_fv is a dict {unrankable feature: fv's}
    # df is the encoded but NOT categorized. DO NOT CHANGE df!!!
    if unrankable_fv != None:
        unrankable = list(unrankable_fv.keys())
        feature_val = merge_dict(unrankable_fv, allocation_book)
    else:
        unrankable = None
        feature_val = allocation_book.copy()
    # deal with discrete rankable
    for col in range(df.shape[1]-1):
        if col not in feature_val.keys():
            feature_val[col] = df['X'+str(col)].nunique()
    if use_cluster == True:
        uncategorized_df = df.copy()
    if qcut == False:
        for col_idx in allocation_book.keys():
            discretized_col = pd.cut(df.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
            df['X'+str(col_idx)] = discretized_col
    else:
        for col_idx in allocation_book.keys():
            discretized_col = pd.qcut(df.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
            df['X'+str(col_idx)] = discretized_col        
    # For pure comonotonic classifier
    X = df[colnames[:-1]].to_numpy()
    Y = df[colnames[-1]].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    if use_cluster == False: # use pure comonotonic
        pure_como_classifier = cm.pure_comonotonic(X_train, Y_train, unrankable = unrankable, feature_val = feature_val)
        pure_como_classifier.run()
        Y_predict = pure_como_classifier.predict(X_test)
    else: # use clustered comonotonic
        #uncategorized_x = uncategorized_df[colnames[:-1]].to_numpy()
        cluster_como_classifier = cm.clustered_comonotonic(X_train, Y_train, unrankable = unrankable, feature_val = feature_val,
                                                        uncategorized_df = uncategorized_df, colnames = colnames,
                                                        min_corr = min_corr, random_state = random_state)
        cluster_como_classifier.run_cluster()
        Y_predict = cluster_como_classifier.predict(X_test)
    accuracy = get_accuracy(Y_predict, Y_test)
    return accuracy


# to check which combination of allocations is the best
# i.e. for each continuous feature, determine how many categories we should divide
# the idea is from simulated annealing algorithm
# cont_col is a list of features which are continuous, num_categories_list is a list of all possible number of categories
# suppose cont_col has size m, num_categories_list has size n, the search space has size n^m
def determine_allocation(cont_col, num_categories_list, df, unrankable_fv,
                        colnames, max_itr, temp, anneal_schedule, use_cluster,
                        use_mistaken_accuracy_test, random_state, min_corr = 0.5, qcut = False):
    allocation_book = {f:num_categories_list[0] for f in cont_col}
    #allocation_book = {f:random.choice(num_categories_list) for f in cont_col}
    allocation_history = {}
    accuracy_history = {}
    for itr in range(max_itr):
        allocation_history[itr] = allocation_book
        if use_mistaken_accuracy_test == True:
            old_accuracy = mistaken_accuracy_test(allocation_book, df, unrankable_fv, colnames, min_corr, use_cluster, random_state, qcut)
        else:
            old_accuracy = accuracy_test(allocation_book, df, unrankable_fv, colnames, min_corr, use_cluster, random_state, qcut)
        accuracy_history[itr] = old_accuracy
        
        if (itr+1)%anneal_schedule == 0:
            temp *= 0.9
        update_feature = cont_col[itr%len(cont_col)]
        old_value = allocation_book[update_feature]
        update_value = random.choice(num_categories_list)
        while update_value == old_value:
            update_value = random.choice(num_categories_list)
        candidate = allocation_book.copy()
        candidate[update_feature] = update_value
        if use_mistaken_accuracy_test == True:
            new_accuracy = mistaken_accuracy_test(candidate, df, unrankable_fv, colnames, min_corr, use_cluster, random_state, qcut)
        else:
            new_accuracy = accuracy_test(candidate, df, unrankable_fv, colnames, min_corr, use_cluster, random_state, qcut)
        
        if new_accuracy > old_accuracy:
            allocation_book = candidate.copy()
        else:
            transition_prob = math.exp((new_accuracy - old_accuracy)/temp)
            determinant = np.random.uniform(0, 1)
            if determinant < transition_prob:
                allocation_book = candidate.copy()
    return accuracy_history, allocation_history

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