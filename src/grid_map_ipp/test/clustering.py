import scipy 
from sklearn.neighbors import NearestNeighbors
import numpy as np 

def cluster(adjacency_matrix, gt=1):
    rows = adjacency_matrix.nonzero()[0]
    cols = adjacency_matrix.nonzero()[1]
    members = []
    member = np.ones(len(range(gt+1)))
    centroids = []
    appendc = centroids.append
    appendm = members.append
    while len(member) > gt:
        un, coun = np.unique(cols, return_counts=True)
        centroid = un[np.argmax(coun)]
        appendc(centroid)
        member = rows[cols == centroid]
        appendm(member)
        cols = cols[np.in1d(rows, member, invert=True)]
        rows = rows[np.in1d(rows, member, invert=True)]
    return members, centroids  

data = np.array([[1.5, 2.0], [2.5, 4.5], [3.5, 6.0], [7.0, 3.5],[3.0 , 9.0], [5.5, 5.0], [0.0, 10.0], [5.5, 3.4], [4.0, 8.0]])
nbrs = NearestNeighbors(metric='euclidean', radius=3.5, 
algorithm='kd_tree')            
nbrs.fit(data)    
adjacency_matrix = nbrs.radius_neighbors_graph(data)

print(adjacency_matrix)
members, centroids = cluster(adjacency_matrix)
print("Members", members)
print("Centroids", centroids)
