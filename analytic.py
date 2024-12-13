import random
import scipy
import scipy.stats
import numpy as np

class UnionFind():
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1]*n
        self.components = n
    def find(self, i):
        """
        Find the root of node i
        """
        if (self.parent[i] == i):
            return i
        else:
            k = self.find(self.parent[i])
            self.parent[i] = k # Path compression
            return k
    def union(self, i, j):
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj:
            return # already in the same group
        self.components -= 1
        if (self.rank[ri] > self.rank[rj]): # Union by rank
            self.parent[rj] = ri
        elif (self.rank[rj] > self.rank[ri]):
            self.parent[ri] = rj
        else:
            self.parent[ri] = rj
            self.rank[ri] += 1
    def connected(self):
        return (self.components == 1)
    
class Graph():
    """
    Graph on n vertices
    """
    def __init__(self, n):
        self.uf = UnionFind(n)
        self.edges = set() # edge set
    def add_edge(self, i, j):
        if i == j: # no self edges
            return
        self.edges.add((i,j))
        self.edges.add((j,i))
        self.uf.union(i,j)
    def num_edges(self):
        return len(self.edges)//2 # we want an undirected graph
    def is_connected(self):
        return self.uf.connected()

def sample_once(n,p):
    """
    Get one sample from the reduced variance sampler
    """
    g = Graph(n)
    while not g.is_connected():
        # Add random edge
        g.add_edge(random.randint(0,n-1), random.randint(0,n-1)) 
    return scipy.stats.binom.sf(g.num_edges()-1, n*(n-1)/2, p) 

def sample(n,p,b):
    """
    Take b samples from an n,p random graph. Returns a tuple containing
    the estimate and the estimated standard error
    """
    return [sample_once(n,p) for _ in range(b)]

def evaluate(vals):
    """
    Evaluate values from the naive sampler"""
    ss = len(vals) # sample size
    p = np.mean(vals) # estimate of p
    se = np.std(vals)/np.sqrt(ss)
    hits = ss # every sample is a hit
    return (p,se,hits,ss)
    
def test():
    # Union find tests
    uf = UnionFind(3)
    uf.union(0,1)
    print(uf.connected()) # False
    uf.union(1,2)
    print(uf.connected()) # True
    # Graph tests
    g = Graph(4)
    g.add_edge(0,1)
    g.add_edge(1,2)
    print(g.num_edges()) # 2
    print(g.is_connected()) # False
    g.add_edge(0,3)
    print(g.is_connected()) # True
    print(evaluate(sample(100, 0.02732, 1000)))

if __name__ == "__main__":
    test()