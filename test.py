import analytic
import mh
import naive
import random
import time
import numpy as np

import pandas

class TestFunction():
    def __init__(self, sample, evaluate, name):
        self.sample = sample
        self.evaluate = evaluate
        self.name = name

class Test():
    def __init__(self, function : TestFunction, n, p, b = 10):
        self.function = function
        self.n = n
        self.p = p
        self.b = b
    def run(self):
        start_time = time.process_time()
        # Run the test and report results
        vals = self.function.sample(self.n,self.p,self.b)
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        mu, se, hits, ess = self.function.evaluate(vals)
        return (elapsed_time, mu, se, hits, ess)
    def __str__(self):
        return f"{self.function.name}: n = {self.n}, p = {self.p}, b = {self.b}"

# Global tfs 
naive_tf = TestFunction(naive.sample, naive.evaluate, "Naive")
analytic_tf = TestFunction(analytic.sample, analytic.evaluate, "Analytic")
mh_1 = TestFunction(lambda n,p,b : mh.sample(n,p,b,1), mh.evaluate, "MH1")
mh_rn = TestFunction(lambda n,p,b : mh.sample(n,p,b,np.sqrt(n)), mh.evaluate, "MHRN")
mh_n = TestFunction(lambda n,p,b : mh.sample(n,p,b,n), mh.evaluate, "MHN")
mh_nln = TestFunction(lambda n,p,b : mh.sample(n,p,b,n*np.log(n)/4), mh.evaluate, "MHNLN")
    
def main():
    random.seed(535)
    tests = []
    # Create our list of tests to carry out
    # for 100000 use 9.558303281664848e-05
    times = (10, 100, 1000, 10000)# , 10000)
    probs = (0.0754, 0.02732, 0.005098)# , 0.0007382)
    test_funcs = (naive_tf, analytic_tf, mh_1, mh_rn, mh_n, mh_nln)
    for n,p in zip(times, probs):
        for tf in test_funcs:
            tests.append(Test(tf, n, p, b = 100000))
    print("The test plan is")
    print("\n".join([str(t) for t in tests]))
    print("Tests will be shuffled")
    input("Press enter to continue:")
    print("Continuing...")
    random.shuffle(tests)
    results = []
    for i,t in enumerate(tests):
        random.seed(535)
        print(f"Running test {t}. [{i+1}/{len(tests)}]")
        process_time, mu, se, hits, ess = t.run()
        results.append((t.function.name, t.n, t.p, t.b, mu, se, hits, ess, process_time))
        print("Done")
    print("Done running tests. Writing out...")
    out_df = pandas.DataFrame(results, columns=["Function", "n", "p", "b", "mu", "se", "hits", "ess", "processTime"])
    out_df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()