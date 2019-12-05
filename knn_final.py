import math
import csv
import operator
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions


def KNN():

    allData = pd.read_csv('Data_26and30_electrode.csv')
    allLabel = pd.read_csv('Data_26and30_electrode_label.csv').values.reshape(-1,)

    #pca
    pca = PCA(n_components=2) 
    pca.fit(allData)
    newSet = pca.fit_transform(allData)

    ##data split
    X_train, X_test, y_train, y_test = train_test_split(newSet, allLabel)

    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train = standardScaler.transform(X_train)
    X_test = standardScaler.transform(X_test)
    knn_model = KNeighborsClassifier(n_neighbors=10)
    
    knn_model.fit(X_train, y_train)

    knn_model.score(X_test, y_test)
    y_predict = knn_model.predict(X_test[:1])

    #results visualization
    plot_decision_regions(newSet,allLabel,clf=knn_model,legend=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn model')
    plt.show()

KNN()
