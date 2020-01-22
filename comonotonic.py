# This is the module for comonotonicity 
import numpy as np
import pandas as pd
import copy
import operator
import utils
from sklearn.model_selection import train_test_split

# handle the case of pure comonotonicity
class pure_comonotonic:
    def __init__(self, x_train, y_train, unrankable):
        """
        Features types: discrete rankable; discrete unrankable; continuous
        unrankable should be a list specifying the columns for discrete unrankable features
        x_train should be categorized before passing in
        """
        self.x_train = x_train # x_train is numpy 2d array
        self.y_train = y_train # y_train is numpy 1d array
        self.unrankable = unrankable # a list
        if self.unrankable != None: # get the indices of features used in comonotonicity
            self.como = list(set([i for i in range(len(x_train[0]))]) - set(unrankable))
        else:
            self.como = [i for i in range(len(x_train[0]))]

    def extract_feature_val(self):
        feature_val = {}
        for i, cols in enumerate(self.x_train.T):
            feature_val[i] = max(np.unique(cols))+1
        self.feature_val = feature_val # how many categories for each feature

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
                    if fv not in class_dict.keys():
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
            for i in range(feature_value):
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
        self.get_prior_prob()
        #print("Complete prior probability")
        self.extract_feature_val()
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
            return 0
        
    def predict_single(self, x):
        # deal with a single case
        prob_distribution = self.get_prob_dist_single(x)
        predicted_class = max(prob_distribution.items(), key=operator.itemgetter(1))[0]
        return predicted_class
    
    def get_prob_dist_single(self, x):
        # get the probability distribution of one instance
        prob_distribution = self.prior_prob.copy() # initialize with prior probability
        if self.unrankable != None:
            for c in prob_distribution.keys():
                for f in self.unrankable:
                    fv = x[f] # fv stands for feature value
                    prob_distribution[c] *= self.unrankable_post_prob[f][c][fv]
        for c in prob_distribution.keys():
            intervals = []
            for f in self.como:
                fv = x[f]
                intervals.append(self.como_prob_interval[f][c][fv])
            prob_distribution[c] *= self.interval_intersection(intervals)
        prob_distribution_list = list(prob_distribution.values())
        prob_distribution_list = [i*(10**8) for i in prob_distribution_list]
        for k in prob_distribution.keys():
            prob_distribution[k] = prob_distribution[k]*(10**8)/sum(prob_distribution_list) 
        return prob_distribution        
            
    def predict(self, x_test):
        y_predict = []
        for x in x_test:
            y_predict.append(self.predict_single(x))
        return y_predict

class clustered_comonotonic(pure_comonotonic):
    def __init__(self, x_train, y_train, unrankable, uncategorized_df, colnames, min_corr, random_state):
        super(clustered_comonotonic, self).__init__(x_train, y_train, unrankable)
        # compute the correlation coefficient matrix by uncategorized data, include unrankable features
        self.uncategorized_df = uncategorized_df
        self.colnames = colnames
        self.min_corr = min_corr
        self.random_state = random_state
    
    def clustering(self):
        # here we need to make sure that we compute the correlation coefficient matrix by the
        # uncategorized X in the training set
        uncategorized_x = self.uncategorized_df[self.colnames[:-1]].to_numpy()
        uncategorized_y = self.uncategorized_df[self.colnames[-1]].to_numpy()
        x_corr_train, x_corr_test, y_corr_train, y_corr_test = train_test_split(uncategorized_x,
                                                                                uncategorized_y, test_size = 0.2, 
                                                                                random_state = self.random_state)
        del x_corr_test
        del y_corr_train
        del y_corr_test
        rankable_var = np.array([[x_corr_train[row][col] for col in self.como] for row in range(len(x_corr_train))])
        corr_matrix = np.corrcoef(rankable_var.T)
        abs_corr = np.absolute(corr_matrix)
        # need to do clustering by corr_matrix
        distance_matrix = 1 - abs_corr
        cluster_book = utils.cluster_agnes(distance_matrix, 1-self.min_corr)
        self.cluster_book = cluster_book # list of list
    
    def run(self):
        self.get_prior_prob()
        self.extract_feature_val()
        if self.unrankable != None:
            self.get_unrankable_prob()
        self.get_comonotonic_prob_interval()
        self.clustering()
    
    def predict_single(self, x):
        prob_distribution = self.get_prob_dist_single(x)
        predicted_class = max(prob_distribution.items(), key=operator.itemgetter(1))[0]
        return predicted_class
    
    def get_prob_dist_single(self, x):
        # get the probability distribution of one instance
        prob_distribution = self.prior_prob.copy() # initialize with prior probability
        if self.unrankable != None:
            for c in prob_distribution.keys():
                for f in self.unrankable:
                    fv = x[f] # fv stands for feature value
                    prob_distribution[c] *= self.unrankable_post_prob[f][c][fv]
        for c in prob_distribution.keys():
            for cluster in self.cluster_book:
                interval = []
                for f_idx in cluster:
                    f = self.como[f_idx]
                    fv = x[f]
                    interval.append(self.como_prob_interval[f][c][fv])
                prob_distribution[c] *= self.interval_intersection(interval)
        prob_distribution_list = list(prob_distribution.values())
        prob_distribution_list = [i*(10**8) for i in prob_distribution_list]
        for k in prob_distribution.keys():
            prob_distribution[k] = prob_distribution[k]*(10**8)/sum(prob_distribution_list) 
        return prob_distribution
        

class naive_bayes:
    # note that the x_train here SHOULD NOT BE CATEGORIZED, pass in after encoding
    # cont_feature is a list containing indices of continuous features
    def __init__(self, x_train, y_train, cont_feature):
        self.x_train = x_train
        self.y_train = y_train
        self.cont_feature = cont_feature
        self.discrete_feature = list(set([i for i in range(len(x_train[0]))]) - set(cont_feature))
    
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

    # extract the discrete feature value
    def get_discrete_feature_val(self):
        discrete_feature_val = {}
        for i in self.discrete_feature:
            discrete_feature_val[i] = max(np.unique(self.x_train.T[i]))+1
        self.discrete_feature_val = discrete_feature_val # how many categories for each feature

    def get_post_prob(self):
        discrete_post_prob = {} # { feature: {class: {fv: post_prob } } }
        for f in self.discrete_feature:
            feature_dict = {}
            for c in self.class_idx.keys():
                class_dict = {}
                for idx in self.class_idx[c]: # traverse the instances of this class
                    if self.x_train[idx][f] not in class_dict.keys():
                        class_dict[self.x_train[idx][f]] = 1
                    else:
                        class_dict[self.x_train[idx][f]] += 1
                # use Laplacian correction to avoid zero probability problem
                all_fv = [i for i in range(self.discrete_feature_val[f])]
                for fv in all_fv:
                    if fv not in class_dict.keys():
                        class_dict[fv] = 1
                summation = sum(class_dict.values())
                class_dict = {k:v/summation for k,v in class_dict.items()}
                feature_dict[c] = class_dict
            discrete_post_prob[f] = feature_dict
        # guassian distribution for continuous features
        # we store the mean and standard deviation for each feature within each class
        cont_mean_sd = {} # { feature: {class: (mean, sd)} }
        for f in self.cont_feature:
            feature_dict = {}
            for c in self.class_idx.keys():
                values = []
                for idx in self.class_idx[c]:
                    values.append(self.x_train[idx][f])
                mean = np.mean(np.array(values))
                sd = np.std(np.array(values))
                feature_dict[c] = (mean, sd)
            cont_mean_sd[f] = feature_dict
        self.discrete_post_prob = discrete_post_prob
        self.cont_mean_sd = cont_mean_sd

    def run(self):
        self.get_prior_prob()
        self.get_discrete_feature_val()
        self.get_post_prob()

    def predict_single(self, x):
        # deal with a single case
        prob_distribution = self.prior_prob.copy() # initialize with prior probability
        for c in prob_distribution.keys():
            for f in self.discrete_feature:
                fv = x[f] # fv stands for feature value
                prob_distribution[c] *= self.discrete_post_prob[f][c][fv]
        for c in prob_distribution.keys():
            for f in self.cont_feature:
                fv = x[f]
                prob_density = utils.gaussian_pdf(self.cont_mean_sd[f][c],fv)
                prob_distribution[c] *= prob_density
        predicted_class = max(prob_distribution.items(), key=operator.itemgetter(1))[0]
        prob_distribution_list = list(prob_distribution.values())
        prob_distribution_list = [i*(10**24) for i in prob_distribution_list]
        for k in prob_distribution.keys():
            prob_distribution[k] = prob_distribution[k]*(10**24)/sum(prob_distribution_list) 
        return predicted_class, prob_distribution
            
    def predict(self, x_test):
        y_predict = []
        for x in x_test:
            y_predict.append(self.predict_single(x)[0])
        return y_predict