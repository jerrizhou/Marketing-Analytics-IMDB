#%% Cancer import functions
import pandas as pd
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.datasets                import load_iris
from sklearn.model_selection         import train_test_split
from sklearn.naive_bayes             import GaussianNB
from sklearn.naive_bayes             import MultinomialNB
from sklearn.naive_bayes             import ComplementNB
from sklearn                         import tree
from sklearn                         import preprocessing
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.metrics                 import confusion_matrix
from sklearn.neighbors               import NearestNeighbors
from sklearn                         import linear_model
from sklearn.neural_network          import MLPClassifier


pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)
os.chdir('D:\\school\\grad\\2020Fall\\Marketing\\project')

#%% Data Cancer read file into python
df = pd.read_csv('movies.csv')
df['movie_year'] = df.movie_year.str.extract('(.)([0-9][0-9][0-9][0-9])(.)')[[1]].astype(int)
df['movie_duration'] = df.movie_duration.str.extract('([0-9]+)( \w\w\w)')[[0]].astype(float)
df['movie_star'] = df.movie_star.astype(float)
df['movie_votes'] = df.movie_votes.str.replace(',', '').astype(int)
df['movie_gross'] = df.movie_gross.str.extract('(.)([0-9]+\.[0-9]+)(M)')[[1]].astype(float)
df.dtypes

#%% Standardization
scaler = preprocessing.StandardScaler().fit(dta_1.iloc[:,[0,1,2,4]])
cancer_X_scaled = scaler.transform(dta_1.iloc[:,[0,12,4]])
cancer_Y = dta_1['movie_star']

#%% Spliting data
x_train_vali, x_test, y_train_vali, y_test = train_test_split(cancer_X_scaled, 
                                                              cancer_Y, 
                                                              test_size    = 0.1,
                                                              random_state = 0)
x_train, x_vali, y_train, y_vali = train_test_split(x_train_vali,
                                                  y_train_vali,
                                                  test_size = 0.1,
                                                  random_state = 0)

#%% Fitting data-- KNN
for k in range(1,10):
    clf_KNN = KNeighborsClassifier(n_neighbors=k)
    clf_KNN.fit(x_train,y_train)
    y_vali_predict_KNN = clf_KNN.predict(x_vali)
    print(confusion_matrix(y_vali, y_vali_predict_KNN))
## not much of a difference, decide to choose 1

 
#%%  Fitting data-- Naive Bayes Classification
clf_NB = GaussianNB().fit(x_train, y_train)
y_vali_predict_NB = clf_NB.predict(x_vali)
print(confusion_matrix(y_vali, y_vali_predict_NB))

#%% Fitting data-- Tree
## using gini
criterion_chosen     = ['entropy','gini'][1]
max_depth            = 10
for depth in range(2,max_depth):
    clf_Tree    = tree.DecisionTreeClassifier(
            criterion    = criterion_chosen, 
            max_depth    = depth,
            random_state = 96).fit(x_train, y_train)
    y_vali_predict_Tree = clf_Tree.predict(x_vali)
    print(confusion_matrix(y_vali, y_vali_predict_Tree))
 ## all parameters preform very well   
    
#%% Fitting data-- Lasso
for alpha in (0.01,0.1,1,10,100):
    clf_Lasso = linear_model.Lasso(alpha= alpha)
    clf_Lasso.fit(x_train, y_train)
    y_vali_predict_Lasso = clf_Lasso.predict(x_vali).round()
    print(confusion_matrix(y_vali, y_vali_predict_Lasso))
## 0.01 performs very well, choose 0.01
        
#%% Fitting data-- nural network
clf_NN = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf_NN.fit(x_train, y_train)
y_vali_predict_NN = clf_NN.predict(x_vali)
print(confusion_matrix(y_vali, y_vali_predict_NN))

clf_NN_3 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 5, 2), random_state=1)
clf_NN_3.fit(x_train, y_train)
y_vali_predict_NN_3 = clf_NN_3.predict(x_vali)
print(confusion_matrix(y_vali, y_vali_predict_NN_3))
## both network work well, choose 2 
#%% Testing data
print('KNN')
clf_KNN = KNeighborsClassifier(n_neighbors=1)
clf_KNN.fit(x_train,y_train)
y_test_predict_KNN = clf_KNN.predict(x_test)
print(confusion_matrix(y_test, y_test_predict_KNN))

print('Naive Bayes')
clf_NB = GaussianNB().fit(x_train, y_train)
y_test_predict_NB = clf_NB.predict(x_test)
print(confusion_matrix(y_test, y_test_predict_NB))

print('Tree')
criterion_chosen     = ['entropy','gini'][1]
clf_Tree    = tree.DecisionTreeClassifier(
        criterion    = criterion_chosen, 
        max_depth    = 2,
        random_state = 96).fit(x_train, y_train)
y_test_predict_Tree = clf_Tree.predict(x_test)
print(confusion_matrix(y_test, y_test_predict_Tree))

print('Lasso')
clf_Lasso = linear_model.Lasso(alpha=0.01)
clf_Lasso.fit(x_train, y_train)
y_test_predict_Lasso = clf_Lasso.predict(x_test).round()
print(confusion_matrix(y_test, y_test_predict_Lasso))

print('Nural Network')
clf_NN = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf_NN.fit(x_train, y_train)
y_test_predict_NN = clf_NN.predict(x_test)
print(confusion_matrix(y_test, y_test_predict_NN))

## looks like all the models except Nural network are preforming very well.