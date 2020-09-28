'''
Conclusion: Using KDTree is really fast. Don't need to worry about nearest neighbor checking 
'''
import numpy as np 
import time
from sklearn.neighbors import *

rng = np.random.RandomState(0)

X = rng.random_sample((10, 2))
print(X)
time_start = time.clock() 
tree = KDTree(X, leaf_size=10) 
print("Build time: ", time.clock() - time_start)
time_start = time.clock() 
dist, ind = tree.query(X[:1], k=10)
# print(dist)
print("Query time: ", time.clock() - time_start)
print(dist)