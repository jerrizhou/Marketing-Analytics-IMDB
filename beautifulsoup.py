# %%%% Preliminaries and library loading
import datetime
import os
import pandas as pd
import re
import shelve
import time
import datetime
import requests

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
from sklearn.metrics                 import accuracy_score

# libraries to crawl websites
from bs4 import BeautifulSoup
from selenium import webdriver
#from pynput.mouse import Button, Controller


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)


# driver = webdriver.Chrome('C:/Users/jerri/Downloads/chromedriver87.exe')


# %%% 


#Creating the list of links.
# links_to_scrape = ['https://www.imdb.com/list/ls056549735/']
# one_link = links_to_scrape[0]
# driver.get(one_link)
link = ['https://www.imdb.com/list/ls056549735/',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=2',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=3',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=4',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=5',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=6',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=7',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=8',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=9',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=10',
            'https://www.imdb.com/list/ls056549735/?sort=list_order,asc&st_dt=&mode=detail&page=11']
            


# %%% 
# Finding all the reviews in the website and bringing them to python
total_movie_review = []
a = 1
for i in link:
    # i = 'https://www.imdb.com/list/ls056549735/'
    reviews           = requests.get(i).content

    soup                         = BeautifulSoup(reviews,'lxml')
            
    movie_list = soup.find_all('div',{'class':'lister-item-content'})
    # r                 = 0
    
    for r in range(len(movie_list)):
        one_review                   = {}
        one_review['scrapping_date'] = datetime.datetime.now()
        # one_review['url']            = driver.current_url
    
    # get the order
        one_review_order = a
        one_review['movie_order']= one_review_order
        
    # get the movie name
        movie_name = movie_list[r].find_all('a')
        if movie_name == []:
            one_review_name = ''
        else:
            one_review_name = movie_name[0].text
        one_review['movie_name']= one_review_name
        
     
     # get the year 
        movie_year = movie_list[r].find_all('span')
        if movie_year == []:
            one_review_year = ''
        else:
            one_review_year = movie_year[1].text
        one_review['movie_year']= one_review_year
        
      # get the classifier 
        movie_class = movie_list[r].find('span', {'class':'certificate'})
        if movie_class is None:
            one_review_class = ''
        else:
            one_review_class = movie_class.text
        one_review['movie_classifier']= one_review_class     
        
    # get the duration 
        movie_dur = movie_list[r].find('span', {'class':'runtime'})
        if movie_dur is None:
            one_review_dur = ''
        else:
            one_review_dur = movie_dur.text
        one_review['movie_duration']= one_review_dur
        
     # get the review 
        movie_review = movie_list[r].find_all('p')
        if movie_review == []:
            one_review_review = ''
        else:
            one_review_review = movie_review[1].text
        one_review['movie_review']= one_review_review
        
    # get genre   
        movie_genre = movie_list[r].find('span', {'class':'genre'})
        try:
            one_review_genre = movie_genre.text
        except IndexError:
            one_review_genre = ''
        except AttributeError:
            one_review_genre = ''    
        one_review['movie_genre']= one_review_genre
    
     # get the star
        movie_star = movie_list[r].find('span', {'class':'ipl-rating-star__rating'})
        try:
            one_review_stars = movie_star.text
        except IndexError:
            one_review_stars = ''
        except AttributeError:
            one_review_genre = ''
        one_review['movie_star']= one_review_stars    
  
    # get the director
        movie_director = movie_list[r].find_all('a')
        try:
            one_review_director = movie_director[12].text 
        except IndexError:
            one_review_director = ''
        except AttributeError:
            one_review_genre = ''
        one_review['movie_director']= one_review_director         
  
    # get the votes
        movie_votes = movie_list[r].find('span', {'name':'nv'})
        try:
            one_review_votes = movie_votes.text
        except IndexError:
            one_review_votes = ''
        except AttributeError:
            one_review_genre = ''
        one_review['movie_votes']= one_review_votes 

    # get the gross
        movie_gross = movie_list[r].find_all('span')
        try:
            one_review_gross = movie_gross[62].text 
        except IndexError:
            one_review_gross = ''
        except AttributeError:
            one_review_genre = ''
        one_review['movie_gross']= one_review_gross
        
        
        total_movie_review.append(one_review)
        a += 1
# %%%% More cleaning
a = pd.DataFrame.from_dict(total_movie_review)
# BeautifulSoup(a.review_raw.iloc[0]).text
a.to_excel('movies.xlsx')

df = pd.DataFrame(total_movie_review)
df.head(10)
df['movie_classifier']
df['movie_director']

#%%
df['movie_year'] = df.movie_year.str.extract('(.)([0-9][0-9][0-9][0-9])(.)')[[1]].astype(int)
df['movie_duration'] = df.movie_duration.str.extract('([0-9]+)( \w\w\w)')[[0]].astype(float)
df['movie_star'] = df.movie_star.astype(float)
df['movie_votes'] = df.movie_votes.str.replace(',', '').astype(int)
df['movie_gross'] = df.movie_gross.str.extract('(.)([0-9]+\.[0-9]+)(M)')[[1]].astype(float)
df.dtypes

#%%
df['movie_star'] = df['movie_star'].round()
dta_1 = df.drop(['scrapping_date', 'movie_name','movie_classifier','movie_review','movie_genre','movie_director','movie_gross'], axis = 1)
dta_1 = dta_1. dropna()

scaler = preprocessing.StandardScaler().fit(dta_1.iloc[:,[0,1,2,4]])
X_scaled = scaler.transform(dta_1.iloc[:,[0,1,2,4]])
Y = dta_1['movie_star']

#%%
x_train_vali, x_test, y_train_vali, y_test = train_test_split(X_scaled, 
                                                              Y, 
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
    accuracy_test_KNN = accuracy_score(y_vali, y_vali_predict_KNN)
    print(confusion_matrix(y_vali, y_vali_predict_KNN), accuracy_test_KNN)
## choose 1
 
#%%  Fitting data-- Naive Bayes Classification
clf_NB = GaussianNB().fit(x_train, y_train)
y_vali_predict_NB = clf_NB.predict(x_vali)
accuracy_test_NB = accuracy_score(y_vali, y_vali_predict_NB)
print(confusion_matrix(y_vali, y_vali_predict_NB), accuracy_test_NB)
## how accuracy on Naive Bayes

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
    accuracy_test_Tree = accuracy_score(y_vali, y_vali_predict_Tree)
    print(confusion_matrix(y_vali, y_vali_predict_Tree),accuracy_test_Tree)
 ## max_depth = 7 preforms the best  
    
#%% Fitting data-- Lasso
for alpha in (0.01,0.1,1,10,100):
    clf_Lasso = linear_model.Lasso(alpha= alpha)
    clf_Lasso.fit(x_train, y_train)
    y_vali_predict_Lasso = clf_Lasso.predict(x_vali).round()
    accuracy_test_Lasso = accuracy_score(y_vali, y_vali_predict_Lasso)
    print(confusion_matrix(y_vali, y_vali_predict_Lasso),accuracy_test_Lasso)
## 0.01 performs very well, choose 0.01
        
#%% Fitting data-- nural network
clf_NN = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf_NN.fit(x_train, y_train)
y_vali_predict_NN = clf_NN.predict(x_vali)
accuracy_test_NN_1 = accuracy_score(y_vali, y_vali_predict_NN)
print(confusion_matrix(y_vali, y_vali_predict_NN),accuracy_test_NN_1)


clf_NN_2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 5, 2), random_state=1)
clf_NN_2.fit(x_train, y_train)
y_vali_predict_NN_2 = clf_NN_2.predict(x_vali)
accuracy_test_NN_2 = accuracy_score(y_vali, y_vali_predict_NN_2)
print(confusion_matrix(y_vali, y_vali_predict_NN),accuracy_test_NN_2)

## choose 2 
#%% Testing data
print('KNN')
clf_KNN = KNeighborsClassifier(n_neighbors=1)
clf_KNN.fit(x_train,y_train)
y_test_predict_KNN = clf_KNN.predict(x_test)
accuracy_test_KNN = accuracy_score(y_test, y_test_predict_KNN)
print(confusion_matrix(y_test, y_test_predict_KNN),accuracy_test_KNN)

print('Naive Bayes')
clf_NB = GaussianNB().fit(x_train, y_train)
y_test_predict_NB = clf_NB.predict(x_test)
accuracy_test_NB = accuracy_score(y_test, y_test_predict_NB)
print(confusion_matrix(y_test, y_test_predict_NB),accuracy_test_NB)

print('Tree')
criterion_chosen     = ['entropy','gini'][1]
clf_Tree    = tree.DecisionTreeClassifier(
        criterion    = criterion_chosen, 
        max_depth    = 7,
        random_state = 96).fit(x_train, y_train)
y_test_predict_Tree = clf_Tree.predict(x_test)
accuracy_test_Tree = accuracy_score(y_test, y_test_predict_Tree)
print(confusion_matrix(y_test, y_test_predict_Tree),accuracy_test_Tree)

print('Lasso')
clf_Lasso = linear_model.Lasso(alpha=0.01)
clf_Lasso.fit(x_train, y_train)
y_test_predict_Lasso = clf_Lasso.predict(x_test).round()
accuracy_test_Lasso = accuracy_score(y_test, y_test_predict_Lasso)
print(confusion_matrix(y_test, y_test_predict_Lasso),accuracy_test_Lasso)

print('Nural Network')
clf_NN = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf_NN_2.fit(x_train, y_train)
y_test_predict_NN_2 = clf_NN_2.predict(x_test)
accuracy_test_NN_2 = accuracy_score(y_test, y_test_predict_NN_2)
print(confusion_matrix(y_test, y_test_predict_NN_2),accuracy_test_NN_2)

## Tree works very well!