# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:55:04 2021

@author: Gabriel A. Morales Ruiz
"""
import pandas as pd
import numpy as np

# Realizar código en Python que reciba una tabla de datos numéricos organizados
# en 2 columnas (x,y) y devuelva los centroides de k>=2 clases usando k-medias.
# Inicialice los centroides con valores aleatorios (de forma uniforme en el
# rango de valores de cada columna).
# Luego, realice código que reciba una pareja de valores (x,y) y devuelva a
# qué clase pertenece de acuerdo a la clasificación obtenida anteriormente

def k_means(data, k = 2, centroids = None, max_iters = 100) :
    """
    K-means algorithm for n dimensions and k clusters/centroids.
    Parameters
    ----------
    data : np.matrix
        Data to classify.
    k : int, optional
        Amount of clusters/centroids. The default is 2.
    centroids : list of lists / np.matrix, optional
        DESCRIPTION. Starting centroids.
    max_iters : int, optional
        Maximum iterations. The default is 100.

    Returns
    -------
    centroids : np.matrix
        Centroids after k-means algorithm finishes.

    """
    if centroids == None :
        centroids = get_random_centroids(data, k)
        
    elif len(centroids) != k :
        AssertionError("Número de centroides no equivale a k")
        
    for i in range(max_iters) :
        old_centroids = centroids
                
        assigned_centroids = assign_centroids(data, centroids)
        
        # Sum the data by cluster
        centroids = [[0]*data.shape[1]] * k
        values_in_centroid = [0] * k

        for i in range(k) :
            assigned_centroids_aux = np.column_stack([assigned_centroids]*data.shape[1])==i
            centroids[i] = np.sum(np.multiply(data, assigned_centroids_aux), axis = 0)
            values_in_centroid[i] = np.sum(assigned_centroids == i)

        # Mean
        for i in range(k) :
            if values_in_centroid[i] > 0 :
                centroids[i] = centroids[i]/values_in_centroid[i]

        centroids = np.stack(centroids, axis=0)

        error = sum([np.linalg.norm(centroids[i] - old_centroids[i]) for i in range(k)])
        if(error < 1e-5) : break
        
    
    return centroids

def get_random_centroids(data, k) :
    """
    Function generates random starting centroids according to the input data's 
    factors ranges.
    Parameters
    ----------
    data : numpy.matrix
        Input data
    k : int
        Amount of clusters/centroids.

    Returns
    -------
    np.matrix
        Randomly generated centroids.

    """
    centroids = []
    columns = np.size(data, axis=1)
    ranges = []
    for i in range(columns) :
        ranges.append([np.min(data[:,i]), np.max(data[:,i])])
    
    for i in range(k) :
        centroid = []
        for span in ranges :
            centroid.append(np.random.uniform(span[0], span[1]))
        centroids.append(centroid)
        
    return np.matrix(centroids)
        
def assign_centroids(data, centroids) :
    """
    Function will calculate the data's distance to the centroids and return
    the index of the pertinent centroid.
    Parameters
    ----------
    data : np.matrix
        Data to classify.
    centroids : np.matrix
        Centroids to use in calculations.

    Returns
    -------
    assigned_centroids : list
        Index of pertinent centroid per row in data.

    """
    distances = []
    for centroid in centroids :
        distances.append(np.linalg.norm(data - centroid, axis=1))
    distances = np.column_stack(distances)
    assigned_centroids = np.argmin(distances, axis=1).flatten()
    return assigned_centroids
    
    
data = pd.read_excel("kmeans_test.xlsx")
data = np.matrix(data.iloc[:,:])

centroids = [[10, 35], [90, 65]]
centroids = k_means(data, centroids = centroids)
# centroids = k_means(data, k = 4)
# print(assign_centroid([[1, 2], [90, 90]], centroids))
