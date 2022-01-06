#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:27:05 2020

@author: Daniel Chen

K-MEANS CLUSTERING ROUND II 

"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from sklearn.preprocessing import Imputer
import seaborn as sns
import csv
import torch #added 12/25/21

#set font size of labels on matplotlib plots
plt.rc('font', size=16)

#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)

def reformatData(data) :
    #PREPROCESSING of "data" array
    #Kite code to remove NaNs
    #https://www.kite.com/python/answers/how-to-remove-nan-values-from-a-numpy-array-in-python

    nans = np.isnan(data)
    not_nans = ~nans
    newData = data[not_nans]

    print("Any NaNs? : ", np.any(np.isnan(newData)))
    print("Is the data all finite? ",np.isfinite(newData.all()))
    print("Does the data have any null values? ", pd.isnull(newData).sum() > 0)
    
    #Assertion tests from StackOverflow
    #https://stackoverflow.com/questions/38351778/conversion-from-numpy-array-float32-to-numpy-array-float64
    assert not np.any(np.isnan(newData))
    assert np.all(np.isfinite(newData))
    assert np.all(newData <= np.finfo('float64').max)
    assert np.all(newData >= np.finfo('float64').min)

    #Imputer code from https://www.kaggle.com/c/word2vec-nlp-tutorial/discussion/11266
    #Not sure why it was necessary. I kept getting this error:
    #ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    data = Imputer().fit_transform(data)

    return data

def pca(data, noComponents) :
    print("\nPCA Dimensionality Reduction")
    pca = PCA(n_components=noComponents)
    pca.fit(data)
    transformed_data = pca.transform(data)
    print("OG shape: ", data.shape)
    print("transformed shape: ", transformed_data.shape)
    return transformed_data

def getLabels(labels_file) :
    labels = []
    with open(labels_file) as csvfile :
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(row[0])
    return labels

if __name__ == "__main__" :

    #Use NumPy helper function "genfromtxt to load data from text file into NumPy arrays.
    #The KMeans class in scikit-learn REQUIRES an np array as an argument.
    '''
    October 2020

    datafile = "r_final_embdgs.csv"
    labels_file = "r_final_labels.csv"

    data = np.genfromtxt(data_file, delimiter = ",")
    data = reformatData(data)
    '''

    model_name = input("FinBERT, LatinBERT, or mBERT?: ")
    if model_name == "FinBERT" :
        data_file = "finbert_adpositions_only/pp_x_tensor.pt"
        labels_file = "finbert_adpositions_only/pp_final_labels.csv"
    elif model_name == "mBERT" :
        data_file = "mbert_adpositions_only/pp_x_tensor.pt"
        labels_file = "mbert_adpositions_only/pp_final_labels.csv"
    labels = getLabels(labels_file)

    tensor = torch.load(data_file)
    data = tensor.cpu().detach().numpy() #SOURCE: https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
    
    #reshape data, remove the dimension that has the weird 6 (1th dimension)
    if ui == "FinBERT" :
        print("Reshaping output of MaskedLMModel (FinBERT)... ")
        print("original data shape: ", data.shape)
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        print("new data shape: ", data.shape)

    data1 = pca(data, 100)

    #SPECIFY NUMBER OF CLUSTERS HERE
    no_clusters = int(input("Number of clusters?: "))

    kmeans = KMeans(n_clusters = no_clusters, max_iter=100)
    kmeans.fit(data1)

    print("Number of iterations: ",kmeans.n_iter_)
    #print(kmeans.cluster_centers_)
    print("Cluster labels for each point: ", kmeans.labels_)

    #Getting the indices of points for each cluster: (https://stackoverflow.com/questions/32232067/cluster-points-after-kmeans-clustering-scikit-learn)
    clusteredPts = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    print("clusteredPts: ",clusteredPts)
    print("length of clusteredPts: ", len(clusteredPts))

    #Output original k-means data file:
    with open("cluster_data_100d_" + str(no_clusters) + "cl.csv", "w") as output0:
        writer = csv.writer(output0)
        for d in data:
            writer.writerow(d)

    #Output cluster labels (adpositions and case marker tokens)
    with open("cluster_assignments_100d_" + str(no_clusters) + "cl.csv", "w") as output :
        writer = csv.writer(output)
        for cluster,indices in clusteredPts.items() :
            theRow = ["","Cluster "+str(cluster)]
            for i in indices :
                word = labels[i] + "(" + str(i) + ")"
                theRow.append(word)

            writer.writerow(theRow)

    #Plot in 2D
    data2 = pca(data, 100)

    kmeans2 = KMeans(n_clusters = no_clusters, max_iter=100)
    kmeans2.fit(data2)
    clusteredPts2 = {i: np.where(kmeans2.labels_ == i)[0] for i in range(kmeans2.n_clusters)}

    with open("cluster_data_2d_" + str(no_clusters) + "cl.csv", "w") as output0:
        writer = csv.writer(output0)
        for d in data2:
            writer.writerow(d)

    fig,zxes = plt.subplots(figsize=(50,50))
    with open("cluster_assignments_2d_" + str(no_clusters) + "cl.csv", "w") as output :
        writer = csv.writer(output)

        for cluster,indices in clusteredPts2.items() :
            theRow = ["","Cluster "+str(cluster)]
            for i in indices :
                word = labels[i] + "(" + str(i) + ")"
                theRow.append(word)
                plt.annotate(word, (data2[i][0],data2[i][1]), size=40)

            writer.writerow(theRow)

    centers = kmeans2.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

    plt.savefig("cluster_assignments_2D_"+ str(no_clusters) + "cl.png")