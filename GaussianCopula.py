import numpy as np
import pandas as pd
from scipy import stats
import copy
import collections
import utils
import operator

class CopulaClassifier:
    """
    Take dependence structure into consideration in order to improve naive bayes
    Using the idea of copulas
    Parameters:
    cont_col: a list containing the indices of continuous variables
    unrankable: a list containing the indices of categorical variables
    feature_val: a dictionary. keys are discrete variables (including rankable & unrankable) 
                 values are number of possible values for this variable
    kernel: the type of KDE for the computation of PDF's
    """
    def __init__(self, cont_col, unrankable, feature_val, min_corr, kernel='gaussian', use_default_bandwidth=True, use_custom_bandwidth=False):
        self.cont_col = cont_col
        self.unrankable = unrankable
        self.feature_val = feature_val
        self.min_corr = min_corr
        self.kernel = kernel
        self.use_default_bandwidth = use_default_bandwidth
        self.use_custom_bandwidth = use_custom_bandwidth
    
    def fit(self, x_train, y_train, custom_bandwidth=None):
        self.x_train = x_train
        self.classes = np.sort(np.unique(y_train)).astype(int)
        self.divided_train_set = [x_train[y_train == yi] for yi in self.classes] # divide the training set by classes
        self.prior_prob = {self.classes[i]:np.log(self.divided_train_set[i].shape[0]/x_train.shape[0]) for i in range(len(self.classes))}
        if self.unrankable != None and len(self.unrankable) != 0:
            self.rankable = list(set([i for i in range(len(self.x_train[0]))]) - set(self.unrankable))
        else:
            self.rankable = [i for i in range(len(self.x_train[0]))]
        if self.feature_val != None:
            self.get_discrete_posterior()
            print("Got discrete posterior")
        self.clustering()
        print("Finished clustering")
    
    """
    get the probability distribution of the test set
    """
    def test_prob_dist(self, x_test):
        test_result = self.copula_density(x_test)
        prob_distribution = {i:{c: np.exp(test_result[i][c]) for c in test_result[i].keys()} for i in test_result.keys()}
        return prob_distribution

    """
    make a prediction for the test set
    """
    def predict(self, x_test):
        prob_distribution = self.test_prob_dist(x_test)
        prediction = {i: max(prob_distribution[i].items(), key=operator.itemgetter(1))[0] for i in prob_distribution.keys()}
        return prediction

    """
    compute by copula
    return {index of test data: {class: prob_density}}
    """
    def copula_density(self, x_test):
        testset_rankable_density = self.estimate_density(x_test)
        test_result = {}
        for i in range(len(x_test)):
            prob_distribution = self.prior_prob.copy()
            for class_idx, c in enumerate(self.classes):
                # unrankable, treat as independent when unrankable features exist
                if self.unrankable != None and len(self.unrankable) != 0:
                    for f in self.unrankable:
                        # add up because the post prob is in logarithm
                        prob_distribution[c] += self.discrete_posterior[c][f][x_test[i][f]]
                # for rankable, if one feature forms a cluster, check if it is discrete, if so, use empirical, otherwise use KDE
                for cluster in self.cluster_book:
                    if len(cluster) > 1:
                        density = self.copula_densities_within_cluster(x_test, i, cluster, testset_rankable_density, class_idx)
                        prob_distribution[c] += density
                    # deal with discrete rankable
                    elif len(cluster) == 1 and cluster[0] not in self.cont_col:
                        f = cluster[0]
                        prob_distribution[c] += np.log(utils.ecdf(self.divided_train_set[class_idx].T[f], x_test[i][f]))
                    # deal with continuous
                    else:
                        f = cluster[0]
                        prob_distribution[c] += testset_rankable_density[c][f][i]
            test_result[i] = prob_distribution
        return test_result

    """
    estimate the probability densities of rankable variables
    store into {class: {feature: [estimated density for instances in the test set]}}
    """
    def estimate_density(self, x_test):
        testset_rankable_density = {}
        for i, c in enumerate(self.classes):
            class_dict = {}
            x = self.divided_train_set[i]
            for f in self.rankable:
                x_i = x.T[f]
                kde = stats.gaussian_kde(x_i)
                class_dict[f] = kde(x_test.T[f])
            testset_rankable_density[c] = class_dict
        testset_rankable_density = testset_rankable_density
        return testset_rankable_density

    """
    compute the density using copula. Note that gaussian copula is onlt conducted within clusters
    cluster is a list containing one cluster
    return a float representing the logarithm of joint density of a cluster of features given a certain class index
    """
    def copula_densities_within_cluster(self, x_test, sample_idx, cluster, testset_rankable_density, class_idx):
        sample_x = x_test[sample_idx]
        R = np.corrcoef(x_test[:,cluster].T)
        marginal_density = []
        marginal_cdf = []
        c = self.classes[class_idx]
        for f in cluster:
            marginal_density.append(testset_rankable_density[c][f][sample_idx])
            marginal_cdf.append(utils.ecdf(self.divided_train_set[class_idx].T[f], sample_x[f]))
        # since the densities are in logarithm form, we can add them up
        density = sum(marginal_density) + np.log(utils.copula_func(marginal_cdf, R))
        return density

    """
    It's possible for some discrete variables to have weak dependence with others
    store all of their information in case of being treated as independent {class:{feature:{fv:post_prob}}}
    the posterior probability has been transfered to logarithm
    """
    def get_discrete_posterior(self):
        self.discrete_posterior = {}
        for c in self.classes:
            class_dict = {}
            for f in self.feature_val.keys():
                feature_dict_train = dict(collections.Counter(self.divided_train_set[c].T[f]))
                uncovered_features = set([i for i in range(self.feature_val[f])]) - set(feature_dict_train.keys())
                if len(uncovered_features) != 0:
                    feature_dict = {k:v+1 for k,v in feature_dict_train.items()}
                    for u_f in uncovered_features:
                        feature_dict[u_f] = 1
                    summation = sum(feature_dict.values())
                    feature_prob_dict = {k:np.log(v/summation) for k,v in feature_dict.items()}
                else:
                    summation = len(self.divided_train_set[c])
                    feature_prob_dict = {k:np.log(v/summation) for k,v in feature_dict_train.items()}
                class_dict[f] = feature_prob_dict
            self.discrete_posterior[c] = class_dict
        return self
        
    """
    use clustering to select variables with large correlation
    return a [ [indices for cluster 1], [indices for cluster 2] ... ]. The indices is for all features
    """   
    def clustering(self):
        rankable_var = self.x_train[:,self.rankable]
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
                adjusted_cluster.append(self.rankable[idx])
            adjusted_cluster_book.append(adjusted_cluster)
        self.cluster_book = adjusted_cluster_book