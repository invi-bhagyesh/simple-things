import sys
input = sys.stdin.readline

def small_div(n, thr):
    if n <= thr:
        return -1
    res = n
    d = 1
    while d * d <= n:
        if n % d == 0:
            if d > thr and d < res:
                res = d
            oth = n // d
            if oth > thr and oth < res:
                res = oth
        d += 1
    return res

t = int(input())
for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    
    mn = min(arr)
    
    k2 = 10**18
    good = True
    for x in arr:
        if x == mn:
            continue
        diff = x - mn
        sd = small_div(diff, mn)
        if sd == -1:
            good = False
            break
        if sd < k2:
            k2 = sd
    
    if good and k2 < 10**18:
        print(max(mn, k2))
    else:
        print(mn)
