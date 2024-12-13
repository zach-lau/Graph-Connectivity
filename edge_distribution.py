import random
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = list(range(n))
        self.sizes = [1]*n # only need sizes of roots
        self.components = n
    def find(self, i):
        # Find the parent of i and perform path compression
        if self.parents[i] == i:
            return i
        else:
            j = self.find(self.parents[i])
            self.parents[i] = j # path compression
            return j
    def union(self, i, j):
        a = self.find(i)
        b = self.find(j)
        # now need them to point to each other
        if (a != b):
            self.components -=1 
            if (self.sizes[a] > self.sizes[b]):
                self.parents[b] = a # small point to big (less children to fix)
                self.sizes[a] += self.sizes[b]
            else:
                self.parents[a] = b
                self.sizes[b] += self.sizes[a]

class SmartGraph:
    def __init__(self, n : int, p : float):
        self.n = n
        self.p = p 
        self.neighbours = [[] for i in range(n)] # no neighbours
        # Basic stats
        self._edges = 0
        self._parents = range(n) # Create our union find data structure
        self._n_components = n 
        self._union_find = UnionFind(n)
    def edges(self):
        return self._edges
    def add_edge(self, i, j):
        self.neighbours[i].append(j)
        self.neighbours[j].append(i)
        self._edges += 1
        self._union_find.union(i,j)
    def connected_components(self):
        return self._union_find.components
    def connected(self):
        return self.connected_components() == 1
    def to_string(self):
        ans = "Graph"
        for i, n in enumerate(self.neighbours):
            ans += f"\n{i}: "
            for v in n:
                ans += f"{v}, "
        return ans
    def add_random_edge(self):
        # just add random edges by trial and error should be fine for sparse
        # graphs
        # can we profile this? what is failing?
        success = False
        while not success:
            i = random.randint(0, self.n-1)
            j = random.randint(0, self.n-1)
            if i != j and j not in self.neighbours[i]:
                self.add_edge(i,j)
                success = True

def run_once(n,p):
    g = SmartGraph(n,p)
    # so we need to weight our samples by based on what they represent
    # so should be times (n choose e) divide by p(e)
    count = 0 
    while not g.connected():
        count += 1
        g.add_random_edge()
    # use sf
    # print(count)
    # print(g.to_string())
    return count
    # could we optimize this? # numerical stability?
    # we could use a self balancing binary search tree ro avl tree?

def estimate_edge_dependence(vals):
    m = max(vals)
    sorted_vals = sorted(vals) # so we can count
    # now we just basically count the number that are less than whatever
    ans = [0]*m
    count = 0
    i = 0 # index into ans
    while i < m:
        while sorted_vals[count] <= i:
            count += 1 
        ans[i] = count/len(vals)
        i = i+1
    return (list(range(0, m)),ans)

# There might bemore speed ups lets see
def run_many(n,p,b):
    vals = [run_once(n,p) for _ in range(b)]
    print(f"Mean: {sum(vals)/b}")
    print(f"Sd: {np.sqrt(np.var(vals))}")
    edge_counts = estimate_edge_dependence(vals)
    # Estimate the probability of a graph with e edges being connected
    plt.plot(edge_counts[0], edge_counts[1])
    pmf_vals = scipy.stats.binom.pmf(edge_counts[0], n*(n-1)/2, p) # pmf(k,n,p)
    # plt.plot(edge_counts[0], pmf_vals)
    joint_p = pmf_vals*edge_counts[1] # renormalize
    plt.plot(edge_counts[0], joint_p/max(joint_p))
    N = n*(n-1)/2
    plt.axvline(N*np.log(n)/n) # this is how many we expect at threshold
    plt.axvline(N*p, color="red") # this is the average number of edges
    plt.axvline(n, color="green") # minimum number of edges for connected graph
    # plt.plot(edge_counts[0], pmf_vals)
    plt.show()

if __name__ == "__main__":
    run_many(100, 0.07, 100000)
    # run_many(1000, [0.005, 0.006], 100) # we could generate the whole curve this way
# Could I do this in some library?
