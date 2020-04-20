import numpy as np
import pandas as pd
import comonotonic as cm
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import copy
import utils
import matplotlib.pyplot as plt
import seaborn as sns

filename = "adult.csv"
random_state = 42
df = pd.read_csv("Datasets/"+filename)
colnames = [('X'+str(i)) for i in range(df.shape[1]-1)]
colnames.append('Y')
df.columns = colnames

#data cleaning for adult.csv
df = df[df.X1 != '?']
df = df[df.X6 != '?']
# unrankable features
encoded_df = utils.encode_df(df, [1,5,6,7,8,9,13])


encoded_df.loc[(encoded_df.Y == "<=50K"),'Y'] = 0
encoded_df.loc[(encoded_df.Y == ">50K"),'Y'] = 1

encoded_df.loc[(encoded_df.X4 <= 9),'X3'] = 0
encoded_df.loc[(encoded_df.X4 == 10),'X3'] = 1
encoded_df.loc[(encoded_df.X4 == 11),'X3'] = 2
encoded_df.loc[(encoded_df.X4 == 12),'X3'] = 2
encoded_df.loc[(encoded_df.X4 >= 13),'X3'] = 3

encoded_df = encoded_df.astype('int32')

cont_col = [0,2,4,10,11,12]
unrankable = [1,5,6,7,8,9,13]
discrete_col = [1,3,5,6,7,8,9,13]
discrete_feature_val = {k:encoded_df['X'+str(k)].nunique() for k in discrete_col}

train_size = 0.7
test_size = 1-train_size
X = encoded_df[colnames[:-1]].to_numpy()
Y = encoded_df[colnames[-1]].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = test_size, random_state = random_state)

clf_nb = GaussianNB()
clf_nb.fit(x_train,y_train)
nb_predict = clf_nb.predict(x_test)
nb_prob_dist = clf_nb.predict_proba(x_test)
nb_probs = nb_prob_dist[:,1]
nb_auc = roc_auc_score(y_test, nb_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

clf_c_como = cm.clustered_comonotonic(x_train,y_train,discrete_feature_val,cont_col,unrankable,0.9)
clf_c_como.run_cluster()
c_como_predict = clf_c_como.cluster_predict(x_test)
c_como_prob_dist = list()
for x in x_test:
    c_prob_dist = clf_c_como.get_cluster_prob_dist_single(x)
    c_como_prob_dist.append(list(c_prob_dist.values()))
c_como_prob_dist = np.array(c_como_prob_dist)
c_probs = c_como_prob_dist[:,1]
c_auc = roc_auc_score(y_test, c_probs)
c_fpr, c_tpr, _ = roc_curve(y_test, c_probs)

clf_p_como = cm.pure_comonotonic(x_train,y_train,discrete_feature_val,cont_col,unrankable)
clf_p_como.run()
p_como_predict = clf_p_como.predict(x_test)
p_como_prob_dist = list()
for x in x_test:
    p_prob_dist = clf_p_como.get_prob_dist_single(x)
    p_como_prob_dist.append(list(p_prob_dist.values()))
p_como_prob_dist = np.array(p_como_prob_dist)
p_probs = p_como_prob_dist[:,1]
p_auc = roc_auc_score(y_test, p_probs)
p_fpr, p_tpr, _ = roc_curve(y_test, p_probs)

baseline_probs = [0 for i in range(len(x_test))]
baseline_fpr, baseline_tpr, _ = roc_curve(y_test, baseline_probs)

plt.plot(nb_fpr, nb_tpr, label = "Naive Bayes")
plt.plot(c_fpr, c_tpr, label = "Clustered Comonotonic")
plt.plot(p_fpr, p_tpr, label = "Aggregated Comonotonic")
plt.plot(baseline_fpr, baseline_tpr, label = "Baseline")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

print("nb auc = " + str(nb_auc))
print("cluster como auc = " + str(c_auc))
print("pure como auc = " + str(p_auc))