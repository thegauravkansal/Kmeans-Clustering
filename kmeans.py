import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cal_centroids(k, centroid_shape):
    
    if k is None:
        return "K should not be empty"
    
    if centroid_shape <=0:
        return "centroid dimensions should be greater than 1"
    
    ''' This function will return random cluster center called centroids'''
    c = [[np.random.randint(0,10) for j in range(centroid_shape)] for i in range(k)]
    return c

def clustering(df, k, centroids, cluster_dim):
    '''This function will assign k cluster to each input where its centroids is minimum'''
    
    if df is None:
        return "Dataframe should not be empty"
    
    if k is None:
        return "K should not be empty"
    
    if len(centroids) < k:
        return "centroids points should be same"
    
    if cluster_dim <=0:
        return "cluster dimensions should be greater than 1"
    
    # df_cluster to store the clustering distance
    df_cluster = pd.DataFrame()

    #calculating  distance of each input from the k centroids
    for i in range(k):
        df_cluster["distance_from_cluster_"+str(i+1)] = np.sqrt(
                sum((df[df.columns[col_index]] - centroids[i][col_index])**2 for col_index in range(cluster_dim))
                )

    #assigning input distance to a cluster where its distance is minimum
    df['cluster'] = df_cluster.idxmin(axis=1).map(lambda x: int(x.lstrip('distance_from_cluster_')))
    
    return df

def update_cluster(df):
    '''This function will return the new centroids'''
    
    if df is None:
        return "Dataframe should not be empty"
    
    for i in range(k):
       for j in range(cluster_dim):
           centroids[i][j] = np.mean(df[df['cluster']==i+1][df.columns[i]])
           
    return centroids
    
if __name__ == '__main__':
    
    #creating a dummy dataframe for testing
    df = pd.DataFrame(np.random.randint(0,1000,size=(100000,4)),columns=['x1', 'x2', 'x3', 'x4'])
    
    #cluster_dim
    cluster_dim = df.shape[1]
    
    #taking default cluster as 3    
    k = 3 
    
    #generating k random centroids
    centroids = cal_centroids(k, df.shape[1])
    
    df = clustering(df, k, centroids, cluster_dim)    
    
    while True:
    
        closest_cluster = df['cluster'].copy(deep = True)
        updated_centroids = update_cluster(df)
        
        df = clustering(df, k, updated_centroids, cluster_dim)
        
        if closest_cluster.equals(df['cluster']):
            break