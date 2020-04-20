# This is the module for comonotonicity 
import numpy as np
import pandas as pd
import copy
import operator
import utils
from sklearn.model_selection import train_test_split

# handle the case of pure comonotonicity
class pure_comonotonic:
    
    def __init__(self, x_train, y_train, discrete_feature_val, cont_col, unrankable, auto_discrete = True, allocation_book = None):
        """
        Features types: discrete rankable; discrete unrankable; continuous
        unrankable should be a list specifying the columns for discrete unrankable features
        x_train does not need to be categorized before passing in 
        if auto_discrete is False, you should pass in allocation_book whose keys are cont columns and values are num of classes 
        """
        self.x_train = x_train # x_train is numpy 2d array
        self.y_train = y_train # y_train is numpy 1d array
        self.cont_col = cont_col
        self.unrankable = unrankable # a list
        self.auto_discrete = auto_discrete
        self.allocation_book = allocation_book
        if self.unrankable != None: # get the indices of features used in comonotonicity
            self.como = list(set([i for i in range(len(x_train[0]))]) - set(unrankable))
        else:
            self.como = [i for i in range(len(x_train[0]))]
        cont_feature_val = {}
        for i in range(len(x_train.T)):
            if i in cont_col:
                if auto_discrete == True:
                    cont_feature_val[i] = 8
                else: # in this way, use the allocation_book to discretize
                    cont_feature_val[i] = allocation_book[i]
        if discrete_feature_val != None:
            self.feature_val = utils.merge_dict(cont_feature_val, discrete_feature_val)
        else:
            self.feature_val = cont_feature_val.copy()

    def discretize(self):
        x_transpose = self.x_train.T
        discrete_x = []
        bin_info = {k:None for k in self.cont_col}
        for i, feature in enumerate(x_transpose):
            if i in self.cont_col:
                if self.auto_discrete == True:
                    discretized, bins = utils.auto_discretize(feature)
                    discrete_x.append(discretized)
                    bin_info[i] = bins.copy()
                else:
                    discretized, bins = utils.custom_discretize(feature, self.allocation_book[i])
                    discrete_x.append(discretized)
                    bin_info[i] = bins.copy()
            else:
                discrete_x.append(feature)
        self.x_train = np.array(discrete_x).T.astype(int)
        self.bin_info = bin_info    

    def get_prior_prob(self):
        # get the prior probability and indices of instances for different classes
        prior_prob = {} # key is class, value is the prior probability of this class
        class_idx = {} # key is class, value is a list containing the indices of instances for this class
        for i in range(len(self.y_train)):
            if self.y_train[i] not in class_idx.keys():
                class_idx[self.y_train[i]] = [i]
            else:
                class_idx[self.y_train[i]].append(i)
        prior_prob = {k:len(v)/len(self.y_train) for k,v in class_idx.items()}
        self.prior_prob = prior_prob
        self.class_idx = class_idx
    
    # compute the posterior probability for features of type unrankable and rankable
    # rankable is a boolean variable
    def get_posterior_prob(self, rankable = True):
        post_prob = {} # { feature_idx: { class: {feature_value: posterior_prob} } }
        if rankable == False:
            feature_list = self.unrankable
        else:
            feature_list = self.como
        for f in feature_list: # f stands for feature
            feature_dict = {} # { class: {feature_value: posterior_prob} }
            for c in self.class_idx.keys(): # c stands for class
                class_dict = {} # {feature_value: posterior_prob}
                for idx in self.class_idx[c]: # traverse the instances of this class
                    if self.x_train[idx][f] not in class_dict.keys():
                        class_dict[self.x_train[idx][f]] = 1
                    else:
                        class_dict[self.x_train[idx][f]] += 1
                # use Laplacian correction to avoid zero probability problem
                all_fv = [i for i in range(self.feature_val[f])]
                for fv in all_fv:
                    if fv not in list(class_dict.keys()):
                        class_dict[fv] = 1
                summation = sum(class_dict.values())
                class_dict = {k:v/summation for k,v in class_dict.items()}
                feature_dict[c] = class_dict
            post_prob[f] = feature_dict
        return post_prob

    def get_unrankable_prob(self):
        # call this function if there exists some discrete unrankable features
        # for unrankable features, treat them as conditional independent
        # { feature_idx: { class: {feature_value: posterior_prob} } }
        self.unrankable_post_prob = self.get_posterior_prob(rankable = False)

    def get_prob_interval(self, class_dict, feature_value):
        # class_dict :: {feature_value: posterior_prob}
        if feature_value == 0:
            return [0, class_dict[feature_value]]
        else:
            inf = 0
            sup = 0
            for i in range(int(feature_value)):
                inf += class_dict[i]
            sup = inf + class_dict[feature_value]
            return [inf, sup]
    
    def get_comonotonic_prob_interval(self):
        # call this function to compute the conditional probability of comonotonic features
        # { feature_idx: { class: {feature_value: posterior_prob} } }
        self.rankable_post_prob = self.get_posterior_prob(rankable = True)
        como_var = np.array([[self.x_train[row][col] for col in self.como] for row in range(len(self.x_train))])
        corr_matrix = np.corrcoef(como_var.T)
        corr_sum = [sum([abs(j) for j in corr_matrix[i]]) for i in range(len(corr_matrix))]
        base_feature_idx_como = corr_sum.index(max(corr_sum)) # the index of the base feature in rankable features
        base_feature = self.como[base_feature_idx_como] # the index of the base feature in all features
        #print("Base feature is " + "X" + str(base_feature))
        prob_interval_collection = {} # {feature_idx: { class: {feature_value: [inf, sup]} } }
        for f in self.rankable_post_prob.keys():
            feature_dict = {} # { class: {feature_value: [inf, sup]} }
            for c in self.rankable_post_prob[f].keys():
                class_dict = {} # {feature_value: [inf, sup]}
                for fv in self.rankable_post_prob[f][c].keys():
                    interval = self.get_prob_interval(self.rankable_post_prob[f][c], fv)
                    if f != base_feature:
                        feature_como_pos = self.como.index(f)
                        if corr_matrix[base_feature_idx_como][feature_como_pos] < 0:
                            interval = [1-interval[1], 1-interval[0]]
                    class_dict[fv] = interval
                feature_dict[c] = class_dict
            prob_interval_collection[f] = feature_dict
            #print(f)
        self.como_prob_interval = prob_interval_collection

    def run(self):
        # this function generalizes the member functions above
        self.discretize()
        self.get_prior_prob()
        #print("Complete prior probability")
        if self.unrankable != None:
            self.get_unrankable_prob()
            #print("Complete unrankable probability")
        self.get_comonotonic_prob_interval()
        #print("Complete comonotonic probability")
      
    def interval_intersection(self, intervals): # intervals is a list of list
        infimum = max([interval[0] for interval in intervals])
        supremum = min([interval[1] for interval in intervals])
        if infimum < supremum:
            return (supremum-infimum)
        else:
            # since pure como results in too many zero probability case
            # we record how far is between the largest infimum and smallest supremum
            return (supremum-infimum)
        
    def predict_single(self, x):
        # deal with a single case
        prob_distribution = self.get_prob_dist_single(x)
        predicted_class = max(prob_distribution.items(), key=operator.itemgetter(1))[0]
        return predicted_class
    
    def get_prob_dist_single(self, x):
        cate_x = []
        for i,f in enumerate(x):
            if i in self.cont_col:
                cate_x.append(np.digitize(f,self.bin_info[i]))
            else:
                cate_x.append(f)
        # get the probability distribution of one instance
        prob_distribution = self.prior_prob.copy() # initialize with prior probability
        if self.unrankable != None:
            for c in prob_distribution.keys():
                for f in self.unrankable:
                    fv = cate_x[f] # fv stands for feature value
                    prob_distribution[c] *= self.unrankable_post_prob[f][c][fv]
        backup_prob_dist = prob_distribution.copy()
        for c in prob_distribution.keys():
            intervals = []
            for f in self.como:
                fv = cate_x[f]
                intervals.append(self.como_prob_interval[f][c][fv])
            prob_distribution[c] *= self.interval_intersection(intervals)
        checker = all(value == 0 for value in prob_distribution.values())
        if checker == True:
            summation = sum(list(backup_prob_dist.values()))
            final_distribution = {}
            for k in backup_prob_dist.keys():
                final_distribution[k] = backup_prob_dist[k]/summation 
        else:        
            summation = sum(list(prob_distribution.values()))
            final_distribution = {}
            for k in prob_distribution.keys():
                final_distribution[k] = prob_distribution[k]/summation
        return final_distribution
            
    def predict(self, x_test):
        y_predict = []
        for x in x_test:
            y_predict.append(self.predict_single(x))
        return y_predict

# please note that x_train and y_train should NOT be categorized
# because they will be used for correlation coefficient matrix calculation
class clustered_comonotonic(pure_comonotonic):

    def __init__(self, x_train, y_train, discrete_feature_val, cont_col, unrankable, 
                 min_corr, auto_discrete = True, allocation_book = None):
        super(clustered_comonotonic, self).__init__(x_train, y_train, discrete_feature_val, cont_col, unrankable, auto_discrete, allocation_book)
        self.min_corr = min_corr

    def clustering(self):
        rankable_var = np.array([[self.x_train[row][col] for col in self.como] for row in range(len(self.x_train))])
        corr_matrix = np.corrcoef(rankable_var.T)
        abs_corr = np.absolute(corr_matrix)
        # need to do clustering by corr_matrix
        distance_matrix = 1 - abs_corr
        # list of list, the indices in cluster book are the indices among RANKABLE features
        cluster_book = utils.cluster_agnes(distance_matrix, 1-self.min_corr)
        adjusted_cluster_book = []
        for cluster in cluster_book:
            adjusted_cluster = []
            for idx in cluster:
                adjusted_cluster.append(self.como[idx])
            adjusted_cluster_book.append(adjusted_cluster)
        self.cluster_book = adjusted_cluster_book

    def run_cluster(self):
        self.get_prior_prob()
        if self.unrankable != None:
            self.get_unrankable_prob()
        self.clustering()
        self.discretize()
        self.get_comonotonic_prob_interval()

    def predict_cluster_single(self, x):
        prob_distribution = self.get_cluster_prob_dist_single(x)
        predicted_class = max(prob_distribution.items(), key=operator.itemgetter(1))[0]
        return predicted_class

    def get_cluster_prob_dist_single(self, x):
        # change x to categorical
        cate_x = []
        for i, f in enumerate(x):
            if i in self.cont_col:
                cate_value = np.digitize(f,self.bin_info[i])
                cate_x.append(cate_value)
            else:
                cate_x.append(f)
        # get the probability distribution of one instance
        prob_distribution = self.prior_prob.copy() # initialize with prior probability
        if self.unrankable != None:
            for c in prob_distribution.keys():
                for f in self.unrankable:
                    fv = cate_x[f] # fv stands for feature value
                    prob_distribution[c] *= self.unrankable_post_prob[f][c][fv]
        backup_prob_dist = prob_distribution.copy()
        for c in prob_distribution.keys():
            for cluster in self.cluster_book:
                interval = []
                for f_idx in cluster:
                    fv = cate_x[f_idx]
                    interval.append(self.como_prob_interval[f_idx][c][fv])
                prob_distribution[c] *= self.interval_intersection(interval)
        checker = all(value == 0 for value in prob_distribution.values())
        if checker == True:
            summation = sum(list(backup_prob_dist.values()))
            final_distribution = {}
            for k in backup_prob_dist.keys():
                final_distribution[k] = backup_prob_dist[k]/summation 
            return final_distribution
        else:        
            summation = sum(list(prob_distribution.values()))
            final_distribution = {}
            for k in prob_distribution.keys():
                final_distribution[k] = prob_distribution[k]/summation
            return final_distribution
    
    def cluster_predict(self, x_test):
        y_predict = []
        for x in x_test:
            y_predict.append(self.predict_cluster_single(x))
        return y_predict