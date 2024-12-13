"""
Metropolis hastings for sampling from random graphs for connectedness
"""
import arviz as az
import numpy as np
import random
import scipy

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
    def num_components(self):
        return self.components
    
class Graph():
    """
    Graph with additional functionality for adding and removing random edges
    """
    def __init__(self, n):
        self.n = n
        self.edge_set = set() # for easy lookup when adding
        self.edge_list = [] # for random removal
        self.edge_count = 0
    def add_m_edges(self, m):
        start_count = self.edge_count
        while self.edge_count < start_count + m:
            # Generate random edges
            i, j = random.randint(0, self.n-1), random.randint(0, self.n-1)
            if (i == j):
                continue
            if (i,j) in self.edge_set:
                continue
            self.edge_set.add((i,j))
            self.edge_set.add((j,i))
            self.edge_list.append((i,j))
            self.edge_count += 1
    def remove_m_edges(self, m):
        if len(self.edge_set) < m:
            raise ValueError(f"Cannot remove {m} edges from a graph with\
            {len(self.edges)} edges!")
        for _ in range(m):
            remove_idx = random.randint(0, self.edge_count-1)
            to_remove = self.edge_list[remove_idx]
            self.edge_list[remove_idx] = self.edge_list[-1]
            self.edge_list.pop() # Remove from the end
            self.edge_set.remove(to_remove)
            self.edge_set.remove(to_remove[::-1])
            self.edge_count -= 1
    def num_components(self):
        # Just add all our edges to a union find data structure
        uf = UnionFind(self.n)
        for (i,j) in self.edge_list:
            uf.union(i,j)
        return uf.num_components()

def sample(n,p,b,scale):
    g = Graph(n)
    lower = n-1
    upper = round(n*np.log(n)/2)
    start = random.randint(lower, upper) # inclusive
    g.add_m_edges(start) # Initialize our graph
    vals = np.ndarray(b, dtype=np.float64)
    comps = np.ndarray(b, dtype=np.float64)
    current_edges = start
    current_comp = g.num_components()
    for i in range(b):
        next_edges = round(np.random.normal(current_edges, scale))
        if (next_edges >= lower and next_edges <= upper and next_edges != current_edges):
            # Update
            if (next_edges > current_edges):
                g.add_m_edges(next_edges-current_edges)
            else:
                g.remove_m_edges(current_edges-next_edges)
            current_edges = next_edges
            current_comp = g.num_components()
        current_connected = (current_comp == 1)
        if current_connected:
            w = (upper+1-lower)*scipy.stats.binom.pmf(current_edges, n*(n-1)/2, p)
        else:
            w = 0
        vals[i] = w
        comps[i] = current_comp
    return vals, comps

def evaluate(val_pair):
    vals, comps = val_pair
    ans = np.mean(vals)
    ess = az.ess(vals)
    se = np.std(vals)/np.sqrt(ess)
    hits = sum([int(x > 0) for x in vals])
    return (ans, se, hits, ess)

def test():
    # sample_mh(10,0.3,10)
    # g = Graph(5)
    # g.add_m_edges(3)
    # print(g.edge_list) # 3 edges
    # g.remove_m_edges(2)
    # print(g.edge_list) # 1 edge
    # print(g.edge_set) # double for both directions
    # real mh
    # print(evaluate(sample(10, 0.07544, 10000,10)))
    # print(evaluate(sample(100, 0.02732, 100000, 25)))
    # print(evaluate(sample(1000, 0.005098, 10000, 10)))
    print(evaluate(sample(10000, 0.0007382, 100000, 100)))
    # print(np.mean(sample_mh(100, 0.02732, 100000, 10)))
    
if __name__ == "__main__":
    test()
