"""
This is an instrumented version of mh.py which is used to measure mixingn of the
chain
"""
import numpy as np
import random
import scipy
import pandas

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
    def num_components(self):
        return (self.components)
    
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
            remove_idx = random.randint(0,self.edge_count-1) # pick edge
            to_remove = self.edge_list[remove_idx] # save the edge
            self.edge_list[remove_idx] = self.edge_list[-1] # replace with end
            self.edge_list.pop() # remove end
            self.edge_set.remove(to_remove) # remove from edge set as well
            self.edge_set.remove(to_remove[::-1])
            self.edge_count -= 1
    def is_connected(self):
        # Just add all our edges to a union find data structure
        uf = UnionFind(self.n)
        for (i,j) in self.edge_list:
            uf.union(i,j)
        return uf.connected()
    def num_components(self):
        # Just add all our edges to a union find data structure
        # We have to rebuild this each time because the graph is dynamic
        uf = UnionFind(self.n)
        for (i,j) in self.edge_list:
            uf.union(i,j)
        return uf.num_components()

def sample(n,p,b,scale):
    g = Graph(n)
    lower = n-1
    upper = round(n*np.log(n))
    start = random.randint(lower, upper) # inclusive
    g.add_m_edges(start) # Initialize our graph
    comps = np.ndarray(b, dtype=np.float64)
    edges = np.ndarray(b, dtype=np.float64)
    vals = np.ndarray(b, dtype=np.float64)
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
        if current_comp == 1:
            w = (upper+1-lower)*scipy.stats.binom.pmf(current_edges, n*(n-1)/2, p)
        else:
            w = 0
        # Add our graph to the last n
        vals[i] = w
        comps[i] = current_comp
        edges[i] = current_edges
    return (vals, edges, comps)

# def evaluate(vals):
#     ans = np.mean(vals)
#     ess = az.ess(vals)
#     se = np.std(vals)/np.sqrt(ess)
#     hits = sum([int(x > 0) for x in vals])
#     return (np.mean(vals),se, hits)

def main():
    """
    Instrumented MCMC to evaluate mixing 
    """
    n = 100
    p =  0.02732
    # p = 0.02
    # n = 10
    # p = 0.07544 
    b = 100000
    agg_vals = []
    agg_edges = []
    agg_comps = []
    tests = ((1,np.sqrt(n),n,n*np.log(n)/4),("MH1","MHRN","MHN","MHNLN"))
    for step in tests[0]:
        random.seed(535)
        vals, edges, comps = sample(n,p,b,step)
        agg_vals.append(vals)
        agg_edges.append(edges)
        agg_comps.append(comps)
    
    df_dict = {}
    for i, name in enumerate(tests[1]):
        df_dict[name+"W"] = agg_vals[i]
        df_dict[name+"E"] = agg_edges[i]
        df_dict[name+"C"] = agg_comps[i]
    df = pandas.DataFrame(df_dict)
    df.to_csv(f"mixing_{n}_{b}.csv")


if __name__ == "__main__":
    main()
