# Naive sampler for random graphs
import random
import numpy as np

def connected(g):
    """
    Check if a graph is connected using DFS. Expects g as an adjacency list. 
    Returns a boolean
    """
    # Now we check if the graph is complete using DFS 
    n = len(g)
    visited = [False]*n
    s = [0] # start our stack with just 0
    n_visited = 0
    while len(s) > 0:
        next_node = s.pop()
        if (visited[next_node]):
            continue # nothing to do
        visited[next_node] = True
        n_visited += 1
        s.extend(g[next_node])
    return (n_visited == n)

def gen_graph(n,p):
    """
    Generate an n,p random graph
    """
    # First we construct our graph as an adjacency list
    # Nodes are labelled 0 through n-1
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if (random.random() < p):
                neighbours[i].append(j)
                neighbours[j].append(i)
    return neighbours

def one_sample(n,p):
    """
    Generate a single random graph and test for connectivity

    Return true if connected, false otherwise
    """
    g = gen_graph(n,p)
    return connected(g)

def sample(n,p,b):
    """
    Take b samples from an n,p random graph. Returns a tuple containing
    the estimate and the estimated standard error
    """
    return [int(one_sample(n,p)) for _ in range(b)]

def evaluate(vals):
    """
    Evluate values from the naive sampler"""
    ss = len(vals) # sample size
    p = np.mean(vals) # estimate of p
    se = np.std(vals)/np.sqrt(ss)
    hits = sum(vals)
    return (p,se,hits,ss)

def test():
    random.seed(535)
    print(gen_graph(10, 0.1)) # [[], [], [4], [6], [2, 9], [], [3], [], [], [4]]
    print(connected([[1],[0,2],[1]])) # TRUE
    print(connected([[1],[0,2],[1],[]])) # FALSE
    print(evaluate(sample(10,0.3,10)))

if __name__ == "__main__":
    test()
