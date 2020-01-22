import numpy as np
import pandas as pd
import copy
import comonotonic as cm
from sklearn.model_selection import train_test_split
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


# to test the accuracy when a certain allocation of number of categories is employed 
def accuracy_test(allocation_book, df, unrankable, colnames, min_corr, use_cluster, random_state): 
    # allocation book is {feature_idx: num_of_categories}; unrankable is a list
    # df is the encoded but NOT categorized. DO NOT CHANGE df!!!
    if use_cluster == True:
        uncategorized_df = df.copy()
    df_for_categorize = df.copy()
    for col_idx in allocation_book.keys():
        discretized_col = pd.cut(df_for_categorize.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
        df_for_categorize['X'+str(col_idx)] = discretized_col
    # For pure comonotonic classifier
    X = df_for_categorize[colnames[:-1]].to_numpy()
    Y = df_for_categorize[colnames[-1]].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    if use_cluster == False: # use pure comonotonic
        pure_como_classifier = cm.pure_comonotonic(X_train, Y_train, unrankable = unrankable)
        pure_como_classifier.run()
        Y_predict = pure_como_classifier.predict(X_test)
    else: # use clustered comonotonic
        #uncategorized_x = uncategorized_df[colnames[:-1]].to_numpy()
        cluster_como_classifier = cm.clustered_comonotonic(X_train, Y_train, unrankable = unrankable,
                                                        uncategorized_df = uncategorized_df, colnames = colnames,
                                                        min_corr = min_corr, random_state = random_state)
        cluster_como_classifier.run()
        Y_predict = cluster_como_classifier.predict(X_test)
    accuracy = get_accuracy(Y_predict, Y_test)
    return accuracy


"""
This is actually a mistaken accuracy test because it changes the value of df after the accuracy test
However, after iterations the accuracy of pure comonotonic rises up for adult.csv dataset
"""
def mistaken_accuracy_test(allocation_book, df, unrankable, colnames, min_corr, use_cluster, random_state): 
    # allocation book is {feature_idx: num_of_categories}; unrankable is a list
    # df is the encoded but NOT categorized. DO NOT CHANGE df!!!
    if use_cluster == True:
        uncategorized_df = df.copy()
    for col_idx in allocation_book.keys():
        discretized_col = pd.cut(df.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
        df['X'+str(col_idx)] = discretized_col
    # For pure comonotonic classifier
    X = df[colnames[:-1]].to_numpy()
    Y = df[colnames[-1]].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    if use_cluster == False: # use pure comonotonic
        pure_como_classifier = cm.pure_comonotonic(X_train, Y_train, unrankable = unrankable)
        pure_como_classifier.run()
        Y_predict = pure_como_classifier.predict(X_test)
    else: # use clustered comonotonic
        #uncategorized_x = uncategorized_df[colnames[:-1]].to_numpy()
        cluster_como_classifier = cm.clustered_comonotonic(X_train, Y_train, unrankable = unrankable,
                                                        uncategorized_df = uncategorized_df, colnames = colnames,
                                                        min_corr = min_corr, random_state = random_state)
        cluster_como_classifier.run()
        Y_predict = cluster_como_classifier.predict(X_test)
    accuracy = get_accuracy(Y_predict, Y_test)
    return accuracy


# to check which combination of allocations is the best
# i.e. for each continuous feature, determine how many categories we should divide
# the idea is from simulated annealing algorithm
# cont_col is a list of features which are continuous, num_categories_list is a list of all possible number of categories
# suppose cont_col has size m, num_categories_list has size n, the search space has size n^m
def determine_allocation(cont_col, num_categories_list, df, unrankable,
                        colnames, max_itr, temp, anneal_schedule, use_cluster,
                        use_mistaken_accuracy_test, random_state, min_corr = 0.5):
    #allocation_book = {f:num_categories_list[0] for f in cont_col}
    allocation_book = {f:random.choice(num_categories_list) for f in cont_col}
    allocation_history = {}
    accuracy_history = {}
    for itr in range(max_itr):
        allocation_history[itr] = allocation_book
        if use_mistaken_accuracy_test == True:
            old_accuracy = mistaken_accuracy_test(allocation_book, df, unrankable, colnames, min_corr, use_cluster, random_state)
        else:
            old_accuracy = accuracy_test(allocation_book, df, unrankable, colnames, min_corr, use_cluster, random_state)
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
            new_accuracy = mistaken_accuracy_test(candidate, df, unrankable, colnames, min_corr, use_cluster, random_state)
        else:
            new_accuracy = accuracy_test(candidate, df, unrankable, colnames, min_corr, use_cluster, random_state)
        
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
    
def gaussian_pdf(mean_sd_tuple, x):
    mean = mean_sd_tuple[0]
    sd = mean_sd_tuple[1]
    return (math.exp(-0.5*(( (x-mean) / sd)**2)))/(sd*math.sqrt(2 * math.pi))

# use weighted average of naive bayes and pure comonotonic
def weighted_avg(nb_dist, p_como_dist, w_nb):
    prob_dist = {c: (nb_dist[c]*w_nb+p_como_dist[c]*(1-w_nb)) for c in nb_dist.keys()}
    return max(prob_dist.items(), key=operator.itemgetter(1))[0]