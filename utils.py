import numpy as np
import pandas as pd
import copy
import comonotonic as cm

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
def accuracy_test(allocation_book, df, unrankable): # allocation book is {feature_idx: num_of_categories}; unrankable is a list
    for col_idx in allocation_book.keys():
        discretized_col = pd.cut(df.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
    df['X'+str(col_idx)] = discretized_col
    # For pure comonotonic classifier
    X = df[colnames[:-1]].to_numpy()
    Y = df[colnames[-1]].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    pure_como_classifier = cm.pure_comonotonic(X_train, Y_train, unrankable = unrankable)
    pure_como_classifier.run()
    Y_predict = pure_como_classifier.predict(X_test)
    accuracy = cm.accuracy(Y_predict, Y_test)
    return accuracy

# to check which combination of allocations is the best
# i.e. for each continuous feature, determine how many categories we should divide
def determine_allocation(cont_col, num_categories_list):
    