# This is the module for comonotonicity 
import numpy as np
import pandas as pd
import copy
import operator

# handle the case of pure comonotonicity
class pure_comonotonic():
    def __init__(self, x_train, y_train, unrankable = None):
        """
        Features types: discrete rankable; binary; discrete unrankable
        unrankable should be a list specifying the columns for discrete unrankable features
        """
        self.x_train = x_train # x_train is numpy 2d array
        self.y_train = y_train # y_train is numpy 1d array
        self.unrankable = unrankable # a list
        if self.unrankable != None: # get the indices of features used in comonotonicity
            self.como = list(set([i for i in range(len(x_train[0]))]) - set(unrankable))
        else:
            self.como = [i for i in range(len(x_train[0]))]
    
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
    def get_posterior_prob(self, rankable):
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
                # maybe you need to add some small numbers to prevent zero conditional probability
                class_dict = {k:v/len(self.class_idx[c]) for k,v in class_dict.items()}
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
        
        prob_interval_collection = {} # {feature_idx: { class: {feature_value: [inf, sup]} } }
        for f in self.rankable_post_prob.keys():
            for c in self.rankable_post_prob[f].keys():
                for fv in self.rankable_post_prob[f][c].keys():
                    interval = self.get_prob_interval(self.rankable_post_prob[f][c], fv)
                    if f != base_feature:
                        feature_como_pos = self.como.index(f)
                        if corr_matrix[base_feature_idx_como][feature_como_pos] < 0:
                            interval = [1-interval[1], 1-interval[0]]
                    prob_interval_collection[f][c][fv] = interval
        self.como_prob_interval = prob_interval_collection

    def run(self):
        # this function generalizes the member functions above
        self.get_prior_prob()
        self.get_unrankable_prob()
        self.get_comonotonic_prob_interval()
      
    def interval_intersection(self, intervals): # intervals is a list of list
        infimum = max([interval[0] for interval in intervals])
        supremum = min([interval[1] for interval in intervals])
        if infimum < supremum:
            return (supremum-infimum)
        else:
            return 0
        
    def predict_single(self, x):
        # deal with a single case
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
        predicted_class = max(prob_distribution.items(), key=operator.itemgetter(1))[0]
        return predicted_class
            
    def predict(self, x_test):
        y_predict = []
        for x in x_test:
            y_predict.append(self.predict_single(x))
        return y_predict