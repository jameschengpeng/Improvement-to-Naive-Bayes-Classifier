import numpy as np
import pandas as pd
import copy
import operator
import utils
from sklearn.model_selection import train_test_split
from scipy.stats import norm

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
            discrete_feature_val[i] = int(max(np.unique(self.x_train.T[i]))+1)
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
        scale_record = {k:0 for k in prob_distribution.keys()}
        for c in prob_distribution.keys():
            for f in self.discrete_feature:
                fv = x[f] # fv stands for feature value
                prob_distribution[c] *= self.discrete_post_prob[f][c][fv]
        for c in prob_distribution.keys():
            for f in self.cont_feature:
                fv = x[f]
                mean = self.cont_mean_sd[f][c][0]
                std = self.cont_mean_sd[f][c][1]
                adjusted_fv = (fv - mean)/std
                prob_density = norm.pdf(adjusted_fv)
                if prob_density < 10**(-3):
                    prob_density = 10**(-3)
                prob_distribution[c] *= prob_density

                if prob_distribution[c] < 0.1:
                    scale, scaled_density = utils.scaler(prob_distribution[c])
                    prob_distribution[c] = scaled_density
                    scale_record[c] += scale
        min_scale = min(scale_record.values())
        scale_record = {k:scale_record[k]-min_scale for k in scale_record.keys()}
        
        adjusted_prob_distribution = {k:prob_distribution[k]*10**(-scale_record[k]) for k in prob_distribution.keys()}
        predicted_class = max(adjusted_prob_distribution.items(), key=operator.itemgetter(1))[0]
        summation = sum(list(adjusted_prob_distribution.values()))

        adjusted_prob_distribution = {k:adjusted_prob_distribution[k]/summation for k in adjusted_prob_distribution.keys()}
        return predicted_class, adjusted_prob_distribution
            
    def predict(self, x_test):
        y_predict = []
        for x in x_test:
            y_predict.append(self.predict_single(x)[0])
        return y_predict