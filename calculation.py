import math

def f(n,p):
    if n == 1:
        return 1
    c = [0]*(n+1)
    c[1] = 1
    for k in range(2,n+1):
        c[k] = 1-sum([c[i]*math.comb(k-1,i-1)*(1-p)**(i*(k-i)) for i in range(1,k)])
    return c[-1]

if __name__ == "__main__":
    # print(f(3,0.07544)) # should be 0.016213 = p^3 +3p^2(1-p)
    print(f(10,0.07544))
    print(f(100,0.02732))
    print(f(1000,0.005098))
    # print(f(10000,0.000732))
